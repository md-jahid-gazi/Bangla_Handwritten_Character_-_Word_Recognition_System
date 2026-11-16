from pathlib import Path
import json, os, time
import numpy as np
import tensorflow as tf
import mlflow
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent

# Dataset is already extracted here. No raw/ or extracted/ folders are required.
DATASET_DIR = ROOT / "data" / "BanglaLekha-Isolated"
IMAGE_DIR = DATASET_DIR / "Images"

MODEL_DIR, MLFLOW_DIR = ROOT / "models", ROOT / "mlflow"
MODEL_DIR.mkdir(exist_ok=True); MLFLOW_DIR.mkdir(exist_ok=True)
VALID={'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}

def natural_key(p):
    try: return (0,int(p.name))
    except ValueError: return (1,p.name.lower())

def preprocess(path):
    with Image.open(path) as im:
        # BanglaLekha-Isolated samples have white strokes on a black background.
        # Invert them so training uses the same black-ink/white-background format
        # as the Streamlit drawing canvas.
        im = ImageOps.invert(ImageOps.grayscale(im))
        a = np.asarray(im)
        ys, xs = np.where(a < 245)  # detect dark handwriting after inversion
        if len(xs) == 0:
            raise ValueError(f"No handwriting/foreground detected in image: {path}")
        im = im.crop((
            max(0, int(xs.min()) - 3),
            max(0, int(ys.min()) - 3),
            min(im.width, int(xs.max()) + 4),
            min(im.height, int(ys.max()) + 4),
        ))
        im.thumbnail((56, 56), Image.Resampling.LANCZOS)
        canvas = Image.new("L", (64, 64), 255)
        canvas.paste(im, ((64 - im.width) // 2, (64 - im.height) // 2))
        return np.asarray(canvas, dtype=np.uint8)[..., None]

def build_model(n):
    from tensorflow.keras import layers,Sequential
    return Sequential([layers.Input((64,64,1)),layers.Rescaling(1/255.),
        layers.Conv2D(32,3,padding='same',activation='relu'),layers.MaxPooling2D(),
        layers.Conv2D(64,3,padding='same',activation='relu'),layers.MaxPooling2D(),
        layers.Conv2D(128,3,padding='same',activation='relu'),layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),layers.Dropout(.3),layers.Dense(128,activation='relu'),layers.Dense(n,activation='softmax')])

def main():
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(
            "Dataset Images folder not found. Expected this exact path:\n"
            f"{IMAGE_DIR}\n"
            "Make sure the dataset is extracted as "
            "data/BanglaLekha-Isolated/Images/<class_id>/<image files>."
        )
    folders = sorted(
        [p for p in IMAGE_DIR.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )
    class_names = [p.name for p in folders]
    expected_ids = [str(i) for i in range(1, 85)]
    if class_names != expected_ids:
        missing = sorted(set(expected_ids) - set(class_names), key=int)
        extra = sorted(set(class_names) - set(expected_ids), key=int)
        raise RuntimeError(
            f"Expected dataset class folders 1-84. Missing: {missing}; extra: {extra}"
        )
    print(f"Using dataset: {IMAGE_DIR}")
    print(f"Found {len(class_names)} class folders.")
    paths = []
    targets = []
    class_counts = {}
    for idx, folder in enumerate(folders):
        files = sorted(
            [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VALID]
        )
        if not files:
            raise RuntimeError(f"No supported images found in class folder: {folder}")
        class_counts[folder.name] = len(files)
        paths.extend(files)
        targets.extend([idx] * len(files))
    print(f"Total images: {len(paths)}")
    print("Images per class:", class_counts)
    if len(class_names)<2 or not paths: raise RuntimeError('No class images found.')
    x=np.stack([preprocess(p) for p in paths]); y=np.asarray(targets,dtype=np.int32)
    xtr,xte,ytr,yte=train_test_split(x,y,test_size=.2,random_state=42,stratify=y)
    tf.keras.utils.set_random_seed(42)
    model=build_model(len(class_names)); model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    mlflow.set_tracking_uri(f'sqlite:///{(MLFLOW_DIR/"mlflow.db").as_posix()}'); mlflow.set_experiment('Bangla_Isolated_Character_Training')
    epochs=int(os.getenv('EPOCHS','8')); batch=int(os.getenv('BATCH_SIZE','64'))
    with mlflow.start_run(run_name=os.getenv('MLFLOW_RUN_NAME','bangla_cnn_baseline')):
        mlflow.log_params({'architecture':'3-block CNN','image_size':'64x64','classes':len(class_names),'epochs':epochs,'batch_size':batch,'learning_rate':.001,'images':len(x),'dataset':'BanglaLekha-Isolated','preprocessing':'invert white-on-black dataset to black-on-white; crop and center 64x64'})
        start=time.time()
        model.fit(xtr,ytr,validation_split=.15,epochs=epochs,batch_size=batch,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=3,restore_best_weights=True)],verbose=2)
        loss,acc=model.evaluate(xte,yte,verbose=0)
        mlflow.log_metrics({'test_loss':float(loss),'test_accuracy':float(acc),'training_seconds':time.time()-start})
        model.save(MODEL_DIR/'model.keras')
        labels={'class_names':class_names,'class_to_index':{n:i for i,n in enumerate(class_names)},'index_to_class':{str(i):n for i,n in enumerate(class_names)},'note':'Folder IDs naturally sorted numerically.'}
        (ROOT/'labels.json').write_text(json.dumps(labels,ensure_ascii=False,indent=2),encoding='utf-8')
        mlflow.log_artifact(str(MODEL_DIR/'model.keras')); mlflow.log_artifact(str(ROOT/'labels.json'))
        print(f'Test accuracy: {acc:.4f}; model saved to {MODEL_DIR/"model.keras"}')
if __name__=='__main__': main()
