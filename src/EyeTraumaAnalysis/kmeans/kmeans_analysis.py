import os
import sys
import numpy as np
import pandas as pd
import scipy
import cv2
from matplotlib import pyplot as plt
import src.EyeTraumaAnalysis


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

def reverse_clustered_image(image_path, K=10):
    # Load the image
    summed_image = plt.imread(image_path)
    print(summed_image.shape)

    # Split the image into its component masks
    masks = []
    for ind in range(K):
        mask = np.zeros(summed_image.shape[:2], dtype=np.uint8)
        mask[summed_image[:,:,0] == ind*25] = 255
        masks.append(mask)


    # get the centers
    centers = []
    for mask in masks:
        original_image = src.EyeTraumaAnalysis.Image(image_path)
        image = cv2.cvtColor(original_image.img, cv2.COLOR_BGR2HSV)
        with_mask = cv2.bitwise_and(image, image, mask=~mask)
        # center = np.mean(with_mask, where=with_mask, axis=[0,1])
        center = np.mean(with_mask, axis=(0,1))
        centers.append(center)
    centers = np.uint8(centers)
        # nonzeros = np.nonzero(mask)
        # center = np.mean(nonzeros, axis=1)
        # centers.append(center)
    # sort centers by HSV "value" - aka sort by grayscale
    # centers = centers.argsort()
    # centers = centers[centers[:, 0].argsort()]
    # TODO: find solution to HSV value sorting for centers when reverse engineered

    print(centers.shape)

    # can't make centers a DataFrame until now since needed numpy for opencv in range or numpy comparison
    centers = pd.DataFrame(centers, columns=["H", "S", "V"])

    summed_image = cv2.cvtColor(summed_image, cv2.COLOR_BGR2HSV)
    print(summed_image.shape)

    mins = np.array([np.min(summed_image, where=kmeans_mask.astype(bool), axis=(0,1)) for kmeans_mask in masks])
    maxs = np.max([np.min(summed_image, where=kmeans_mask.astype(bool), axis=(0,1)) for kmeans_mask in masks])
    ranges = pd.DataFrame(maxs - mins, columns=["H", "S", "V"])

    return centers, ranges, masks

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



def calculate_roc(truths, predict_scores, true_value=None, comparator=None):
    if true_value is None:
        truths = np.array(truths, dtype=bool)
    elif isinstance(true_value,(list,dict,tuple,set,pd.Series,pd.DataFrame,np.ndarray)):
        truths = np.array(truths) #== true_value  # check if values equal the whole collection
        truths_temp = np.zeros(truths.shape, dtype=bool)  # start with false and then apply the | (or) operator
        for each_true_value in true_value:  # check if values equal any of the elements in the collection
            truths_temp = truths_temp | (truths == each_true_value)
        truths = truths_temp
    else:
        truths = np.array(truths) == true_value

    if np.all(truths) or ~np.any(truths):
        raise ValueError("All truth values are " + truths[0])

    predict_scores = np.array(predict_scores)
    score_options = np.sort(np.unique(predict_scores))
    #thresholds = np.concatenate( ([np.min(predict_scores)-0.01], predict_scores) )
    thresholds = np.concatenate( (
        [score_options[0]-0.01],
        np.mean([score_options[1:],score_options[:-1]],axis=0), # do np.mean instead of (+)/2 to avoid issues with
        # uint8 data loss after you get values past 255
        [score_options[-1]+0.01]
    ))
    comparator_original = comparator
    if comparator is None:
        # Guess a comparator based off mean values. This is not a sure fire approach and is checked at the end of the
        # function by the AUC.
        scores_of_trues  = np.mean(predict_scores, where=truths)
        scores_of_falses = np.mean(predict_scores, where=~truths)
        if scores_of_trues >= scores_of_falses:
            comparator = "≥"
        else:
            comparator = "≤"

    if comparator in [">=","≥"]:
        predictions = predict_scores >= thresholds[...,np.newaxis]
        comparator_opposite = "≤"
    elif comparator in ["<=","≤"]:
        predictions = predict_scores <= thresholds[...,np.newaxis]
        comparator_opposite = "≥"
    elif comparator in [">"]:
        predictions = predict_scores > thresholds[...,np.newaxis]
        comparator_opposite = "<"
    elif comparator in ["<"]:
        predictions = predict_scores < thresholds[...,np.newaxis]
        comparator_opposite = ">"
    else:
        raise ValueError(f'Comparator "{comparator}" is not one of the valid options: [">=","<=",">","<"]')

    # predictions has one more dimension than predict_scores
    true_pos  =  truths &  predictions
    false_pos = ~truths &  predictions
    false_neg =  truths & ~predictions
    true_neg  = ~truths & ~predictions

    true_pos_ct  = np.count_nonzero(true_pos,  axis=-1)
    false_pos_ct = np.count_nonzero(false_pos, axis=-1)
    false_neg_ct = np.count_nonzero(false_neg, axis=-1)
    true_neg_ct  = np.count_nonzero(true_neg,  axis=-1)

    with np.errstate(invalid="ignore"):
        # Below is a good paper to review the formulas
        # https://www.frontiersin.org/articles/10.3389/fpubh.2017.00307/full
        accuracy = true_pos_ct + true_neg_ct / ( true_pos_ct + false_pos_ct + false_neg_ct + true_neg_ct )
        # Sensitivity aka Recall aka True positive rate (TPR)
        sensitivity = tpr = true_pos_ct / ( true_pos_ct + false_neg_ct )
        # Specificity aka True negative rate (TNR)
        specificity = tnr = true_neg_ct / ( true_neg_ct + false_pos_ct )
        # Positive predictive value (PPV) aka Precision
        ppv = true_pos_ct / ( true_pos_ct + false_pos_ct )
        # Negative predictive value (NPV)
        npv = true_neg_ct / ( true_neg_ct + false_neg_ct )
        # False discovery rate (FDR)
        fdr = 1 - ppv
        # False omission rate (FOR, called FOMR in code)
        fomr = 1 - npv
        # False negative rate (FNR)
        fnr = 1 - tpr
        # False positive rate (FPR) aka 1-specificity
        fpr = 1 - tnr

        roc_df = pd.DataFrame({
            "threshold": thresholds,
            "sensitivity": sensitivity,
            "1-specificity": fpr,
            "specificity": specificity,
        }).sort_values(by="specificity", ascending=False)
        auc = np.trapz(y=roc_df["sensitivity"],x=1-roc_df["specificity"])
    if comparator_original is None and auc <0.5:
        # if no specific comparator was put in and auc<0.5, then switch the comparator to get an auc≥0.5
        return calculate_roc(truths, predict_scores, true_value=None, comparator=comparator_opposite)
    else:
        return roc_df, auc, comparator
