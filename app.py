import os, tempfile, zipfile
from flask import Flask, request, render_template, send_file
import rasterio, numpy as np, geopandas as gpd, pandas as pd
from rasterio.mask import mask

app = Flask(__name__)

def unzip(file_storage):
    td = tempfile.mkdtemp()
    with zipfile.ZipFile(file_storage, 'r') as z:
        z.extractall(td)
    return td

def find(fname_list, key):
    for f in fname_list:
        if key in f.lower():
            return f
    return None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    files = request.files.getlist("images[]")
    shpzip = request.files["shapefile_zip"]
    thr = float(request.form["threshold"])

    img_dir = tempfile.mkdtemp()
    for f in files:
        path = os.path.join(img_dir, os.path.basename(f.filename))
        f.save(path)
    fnames = os.listdir(img_dir)
    red = rasterio.open(os.path.join(img_dir, find(fnames, "red")))
    nir = rasterio.open(os.path.join(img_dir, find(fnames, "nir")))
    green = rasterio.open(os.path.join(img_dir, find(fnames, "green")))
    re = None
    re_fn = find(fnames, "rededge")
    if re_fn:
        re = rasterio.open(os.path.join(img_dir, re_fn))

    profile = red.profile
    arr_r = red.read(1).astype(float)
    arr_n = nir.read(1).astype(float)
    arr_g = green.read(1).astype(float)
    arr_re = re.read(1).astype(float) if re else None

    eps = 1e-6
    ndvi = (arr_n - arr_r) / (arr_n + arr_r + eps)
    ndre = (arr_n - arr_re) / (arr_n + arr_re + eps) if arr_re is not None else None
    gndvi = (arr_n - arr_g) / (arr_n + arr_g + eps)
    osavi = 1.16 * (arr_n - arr_r) / (arr_n + arr_r + 0.16 + eps)
    savi = 1.5 * (arr_n - arr_r) / (arr_n + arr_r + 0.5 + eps)
    grri = arr_g / (arr_r + eps)
    ngrdi = (arr_g - arr_r) / (arr_g + arr_r + eps)
    sr = arr_n / (arr_re + eps) if arr_re is not None else None
    ccci = (ndre / (ndvi + eps)) if ndre is not None else None
    gci = (arr_n / (arr_g + eps)) - 1
    reci = (arr_n / (arr_re + eps)) - 1 if arr_re is not None else None
    ndrer = (arr_re / (arr_r + eps)) if arr_re is not None else None

    idxs = {
      "NDVI": ndvi,
      "NDRE": ndre,
      "GNDVI": gndvi,
      "OSAVI": osavi,
      "SAVI": savi,
      "GRRI": grri,
      "NGRDI": ngrdi,
      "SR": sr,
      "CCCI": ccci,
      "GCI": gci,
      "RECI": reci,
      "NDRER": ndrer
    }
    out_tifs = {}
    profile.update(dtype=rasterio.float32, count=1, nodata=np.nan)
    for name, arr in idxs.items():
        if arr is None: continue
        p = os.path.join(img_dir, f"{name}.tif")
        with rasterio.open(p, 'w', **profile) as dst:
            dst.write(arr.astype(rasterio.float32), 1)
        out_tifs[name] = p

    shpdir = unzip(shpzip)
    shp = [f for f in os.listdir(shpdir) if f.endswith(".shp")][0]
    gdf = gpd.read_file(os.path.join(shpdir, shp))

    rows = []
    for _, row in gdf.iterrows():
        geom = [row.geometry]
        pid = row.get("PlotID", _)
        rec = {"PlotID": pid}
        for name, tif in out_tifs.items():
            with rasterio.open(tif) as src:
                arr, _ = mask(src, geom, crop=True)
            data = arr[0]
            valid = data[~np.isnan(data)]
            if valid.size == 0:
                rec.update({f"{name}_Mean":np.nan, f"{name}_Median":np.nan,
                            f"{name}_Min":np.nan, f"{name}_Max":np.nan,
                            f"{name}_%AboveThr":np.nan})
            else:
                rec[f"{name}_Mean"] = float(valid.mean())
                rec[f"{name}_Median"] = float(np.median(valid))
                rec[f"{name}_Min"] = float(valid.min())
                rec[f"{name}_Max"] = float(valid.max())
                rec[f"{name}_%AboveThr"] = float((valid>thr).sum()/valid.size*100)
        rows.append(rec)

    df = pd.DataFrame(rows)
    ex = os.path.join(img_dir,"veg_indices_stats.xlsx")
    df.to_excel(ex, index=False)
    return send_file(ex, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
