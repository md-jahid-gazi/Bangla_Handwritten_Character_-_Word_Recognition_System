from pathlib import Path
from PIL import Image, ImageDraw
import argparse, math
VALID={".png",".jpg",".jpeg",".bmp",".tif",".tiff"}
p=argparse.ArgumentParser()
p.add_argument("--root",default="data/extracted/BanglaLekha-Isolated/images")
p.add_argument("--out",default="artifacts/contact_sheets")
a=p.parse_args()
root=Path(a.root); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
for folder in sorted([x for x in root.iterdir() if x.is_dir()],key=lambda x:x.name):
    files=[x for x in folder.rglob("*") if x.is_file() and x.suffix.lower() in VALID][:25]
    if not files: continue
    cellw,cellh=120,140; cols=5; rows=math.ceil(len(files)/cols)
    sheet=Image.new("RGB",(cols*cellw,rows*cellh),"white"); d=ImageDraw.Draw(sheet)
    for i,f in enumerate(files):
        im=Image.open(f).convert("L"); im.thumbnail((100,100))
        x=(i%cols)*cellw+(cellw-im.width)//2; y=(i//cols)*cellh+5
        sheet.paste(im,(x,y)); d.text(((i%cols)*cellw+5,(i//cols)*cellh+110),f.name[:18],fill="black")
    sheet.save(out/f"class_{folder.name}.jpg")
print("Contact sheets saved to",out)
