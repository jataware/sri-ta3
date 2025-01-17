import numpy as np
import rasterio as rio
from copy import copy


def write_tif(results, path, attributions_flag, datamodule):
    
    
    
    # defines the tif meta data
    tif_meta = copy(datamodule.data_predict.tif_meta)
    tif_meta.update({
        "count": 1,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 128,
        "blockysize": 128,
    })

    # extracts raster pts and data
    result_pts = np.dot(
        np.asarray((~tif_meta["transform"]).column_vectors).T, 
        np.vstack((results[:,0], results[:,1], np.ones_like(results[:,1])))
    ).astype(int).T
    data = results[:,2:]
    
    output_rasters = [
        "Likelihoods", 
        "Uncertainties",
    ]

    if attributions_flag:
        attrs = [None] * len(datamodule.data_predict.tif_tags)
        for tag, idx in datamodule.data_predict.tif_tags.items(): attrs[int(idx)] = tag
        output_rasters = output_rasters + attrs[:-1] # last tag is label - doesn't exist

    tif_files = []
    for idx, tif_layer in enumerate(output_rasters):
        # forms tif ndarray
        tif_data = np.empty(shape=(tif_meta["height"], tif_meta["width"]))
        tif_data[:] = np.nan
        tif_data[result_pts[:,1], result_pts[:,0]] = data[:,idx].astype(float)
        
        # writes the output tif
        tif_file = f"{path}/{tif_layer}.tif"
        with rio.open(tif_file, "w", **tif_meta) as out:
            out.write_band(1, tif_data)
        tif_files.append(tif_file)
    return tif_files
