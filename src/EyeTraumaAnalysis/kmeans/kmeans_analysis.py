import os
import sys
import numpy as np
import pandas as pd
import scipy
import cv2


def create_kmeans(img_bgr, K=10):  # K is number of clusters
    #np.all(skimage.io.imread("data/01_raw/14579.png") == skimage.io.imread(data_row.row_data))
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    Z_hsv = img_hsv.reshape((-1,3))
    # convert to np.float32
    Z_hsv = np.float32(Z_hsv)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,centers=cv2.kmeans(Z_hsv,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    centers = np.uint8(centers)
    res_hsv = centers[label.flatten()]
    res_hsv2 = res_hsv.reshape(img_hsv.shape)
    res_bgr = cv2.cvtColor(res_hsv2, cv2.COLOR_HSV2BGR)
    # res2 = cv2.cvtColor(res2, cv2.COLOR_RGB2GRAY)


    # sort centers by HSV "value" - aka sort by grayscale
    centers = centers[centers[:, 2].argsort()]

    #centers_indices = np.argsort(centers, axis=0)   # sorts each column separately

    kmeans_masks = []
    for ind in range(K):
        # Can use opencv in range or kmeans
        #kmeans_masks.append(cv2.inRange(res_hsv2, centers[ind], centers[ind]))
        kmeans_masks.append( np.all(res_hsv2 == centers[ind], axis=-1) )
        #kmeans_masks.append( res_hsv2==centers[ind])
    kmeans_masks = np.array(kmeans_masks)

    # can't make centers a DataFrame until now since needed numpy for opencv in range or numpy comparison
    centers = pd.DataFrame(centers, columns=["H","S","V"])
    mins = np.array([np.min(img_hsv[kmeans_mask],axis=0) for kmeans_mask in kmeans_masks])
    maxs = np.max([np.min(img_hsv[kmeans_mask],axis=0) for kmeans_mask in kmeans_masks])
    ranges = pd.DataFrame(maxs - mins, columns=["H","S","V"])
    return centers, ranges, res_bgr, kmeans_masks

def get_spatial_metrics(mask):
    # scipy can perform the mean (center of mass), but not the standard deviation
    # spatial_means = snd.center_of_mass(mask)
    x = np.linspace(0, 1, mask.shape[1])
    y = np.linspace(0, 1, mask.shape[0])
    xgrid, ygrid = np.meshgrid(x, y)
    grids = {"x": xgrid, "y":ygrid}
    to_return = {"x":{}, "y":{}}
    for ind, grid in grids.items():
        to_return[ind]["mean"] = np.mean(grids[ind], where=mask.astype(bool))
        to_return[ind]["sd"] = np.std(grids[ind], where=mask.astype(bool))
    return to_return


def get_kmeans_metrics(centers, ranges, kmeans_masks):
    spatial_metrics_list = [get_spatial_metrics(kmeans_mask) for kmeans_mask in kmeans_masks]
    spatial_metrics_pd = pd.concat([pd.DataFrame({
        "x": [spatial_metrics["x"]["mean"] for spatial_metrics in spatial_metrics_list],
        "y": [spatial_metrics["y"]["mean"] for spatial_metrics in spatial_metrics_list],}),
        pd.DataFrame({
        "x": [spatial_metrics["x"]["sd"] for spatial_metrics in spatial_metrics_list],
        "y": [spatial_metrics["y"]["sd"] for spatial_metrics in spatial_metrics_list],
    })], axis=1, keys=["Mean","SD"])

    area_fractions = pd.DataFrame([np.count_nonzero(kmeans_mask)/np.prod(kmeans_mask.shape) for kmeans_mask in
                               kmeans_masks], columns=pd.MultiIndex.from_tuples([("","")]))
    color_metrics = pd.concat([centers, ranges], axis=1, keys=["Center","Range"])

    all_metrics = pd.concat([color_metrics, spatial_metrics_pd, area_fractions], axis=1,
                            keys=["Color","Location","Area"])
    all_metrics_ranks = np.argsort(all_metrics, axis=0) + 1

    return pd.concat([all_metrics, all_metrics_ranks], axis=1, keys=["Values","Ranks"])

def choose_kmeans_cluster(metrics):
    metrics = metrics.copy()
    metrics[("Values","Location","SD","x y")] = metrics[
        [("Values","Location","SD","x"),
         ("Values","Location","SD","y")]].max(axis=1) # get max of x and y SD
    likely = metrics[
        (metrics["Ranks"]["Color"]["Center"]["V"] >= 5) &
        (metrics["Values"]["Location"]["Mean"]["x"] >= 0.3) &
        (metrics["Values"]["Location"]["Mean"]["x"] <= 0.7) &
        (metrics["Values"]["Location"]["Mean"]["y"] >= 0.3) &
        (metrics["Values"]["Location"]["Mean"]["y"] <= 0.7) &
        (metrics["Values"]["Location"]["SD"]["x"] <= 0.25) &
        (metrics["Values"]["Location"]["SD"]["y"] <= 0.25)
    ]
    # trim down further
    if likely.shape[0] > 2:
        likely = likely.sort_values(by=("Values","Location","SD","x y"))[:2]
    return likely