import nibabel as nb
import nipy
import numpy as np
import matplotlib.pyplot as plt
import nipy.algorithms.registration
from ip_utils2 import *
from scipy import ndimage
import os
from glob import glob
from subprocess import call

def warp_mni_to_img(subcode, target, warp, phy_coords):
    ''' Warp (physical-space) MNI-space coords to a target image. '''
    phy_coords = np.array([[-24.,-1.,-21.]]).T
    warp = nb.load(os.path.join('/hcp',subcode, 'MNINonLinear/xfms/standard2acpc_dc.nii.gz'))
    target = nb.load(os.path.join('/data/hcp/data/',subcode,'/dwi_MD.nii.gz'))
    warp_img_coords = xform_coords(np.linalg.inv(warp.get_affine()), phy_coords)
    warp_data = warp.get_data()
    xo = ndimage.map_coordinates(warp_data[...,0], warp_img_coords, order=1)
    yo = ndimage.map_coordinates(warp_data[...,1], warp_img_coords, order=1)
    zo = ndimage.map_coordinates(warp_data[...,2], warp_img_coords, order=1)
    phy_coords[0,:] -= xo
    phy_coords[1,:] += yo
    phy_coords[2,:] += zo
    warped_img_coords = xform_coords(np.linalg.inv(target.get_affine()), phy_coords)
    return warped_img_coords

def warp_img_to_mni(source, warp, img_coords, mni_ref=None):
    ''' Warp source image coords to mni space. If an mni reference image is passed, 
        then the coordinates returned are in that image space. If mni_ref is none,
        the returned coords are in MNI physical space.
    '''
    phy_coords = xform_coords(source.get_affine(), img_coords)
    warp_img_coords = xform_coords(np.linalg.inv(warp.get_affine()), phy_coords)
    warp_data = warp.get_data()
    xo = ndimage.map_coordinates(warp_data[...,0], warp_img_coords, order=1)
    yo = ndimage.map_coordinates(warp_data[...,1], warp_img_coords, order=1)
    zo = ndimage.map_coordinates(warp_data[...,2], warp_img_coords, order=1)
    phy_coords[0,:] -= xo
    phy_coords[1,:] += yo
    phy_coords[2,:] += zo
    if mni_ref==None:
        return phy_coords
    else:
        return xform_coords(np.linalg.inv(mni_ref.get_affine()), phy_coords)

def xform_sl(subcode, sl_file, img_coords=True):
    out_dir = os.path.join('/data/hcp/data',subcode)
    warp = nb.load(os.path.join('/hcp',subcode,'MNINonLinear/xfms/acpc_dc2standard.nii.gz'))
    mni = nb.load('/usr/share/fsl/data/standard/MNI152_T1_1mm.nii.gz')
    #fa = nb.load(os.path.join(out_dir,'dwi_FA.nii.gz'))
    ref = nb.load(os.path.join('/hcp', subcode, 'T1w/T1w_acpc_dc_restore_1.25.nii.gz'))
    sl = load_streamline_file(os.path.join(out_dir, sl_file))
    if img_coords:
        slx = [warp_img_to_mni(ref, warp, np.array(s).T, mni).T.tolist() for s in sl]
    else:
        slx = [warp_img_to_mni(ref, warp, np.array(s).T).T.tolist() for s in sl]
    return(slx)
    
def make_roi(target, warp, mni_roi, dilation=0):
    sz = target.shape
    # Image space coords of target:
    #x,y,z = np.meshgrid(range(sz[0]),range(sz[1]),range(sz[2]), indexing='ij')
    img_coords = np.array(np.meshgrid(range(sz[0]),range(sz[1]),range(sz[2]), indexing='ij')).reshape((3,-1))
    # Physical space coords of target:
    phy_coords = xform_coords(target.get_affine(), img_coords)
    # MNI image space coords for target 
    mni_img_coords = xform_coords(np.linalg.inv(warp.get_affine()), phy_coords)
    # pull the offsets from the MNI-space LUT
    warp_data = warp.get_data()
    xo = ndimage.map_coordinates(warp_data[...,0], mni_img_coords, order=1)
    yo = ndimage.map_coordinates(warp_data[...,1], mni_img_coords, order=1)
    zo = ndimage.map_coordinates(warp_data[...,2], mni_img_coords, order=1)
    # Apply the offsets to physical space coords of the target
    # FIXME: I think the +/- is related to the affine. These values work for the images 
    # that we are processing (i.e., our results match fsl's applywarp). But I suspect
    # this is not a general solution...
    phy_coords[0,:] -= xo
    phy_coords[1,:] += yo
    phy_coords[2,:] += zo
    # convert the target physical space coords to roi image space
    roi_coords = xform_coords(np.linalg.inv(mni_roi.get_affine()), phy_coords)
    # Pull the values from the ROI map
    roi_vals = ndimage.map_coordinates(mni_roi.get_data(), roi_coords, order=1)
    roi_vals = roi_vals.reshape(sz)
    if dilation>0:
        roi_vals = ndimage.binary_dilation(roi_vals, iterations=dilation)
    roi_vals = ndimage.binary_fill_holes(roi_vals).astype(np.int8)
    roi = nb.Nifti1Image(roi_vals, target.get_affine())
    return roi

def make_rois(subcode, roi_names, roi_dilation=None, qa=False):
    out_dir = os.path.join('/data/hcp/data',subcode)
    sub_dir = os.path.join('/hcp',subcode)
    mni_lut = nb.load(os.path.join(sub_dir,'MNINonLinear/xfms/standard2acpc_dc.nii.gz'))
    #t1 = nb.load(os.path.join(sub_dir,'T1w/T1w_acpc_dc.nii.gz')
    #ref = nb.load(os.path.join(out_dir,'dwi_FA.nii.gz'))
    ref = nb.load(os.path.join(sub_dir,'T1w', 'Diffusion', 'nodif_brain_mask.nii.gz'))
    for i,roi_name in enumerate(roi_names):
        mni_roi = nb.load('/data/ROIs/' + roi_name + '.nii.gz')
        if roi_dilation!=None and len(roi_dilation)>i:
            dilation = roi_dilation[i]
            print('Dilating %s by %d...' % (roi_name, dilation))
        roi = make_roi(ref, mni_lut, mni_roi, roi_dilation)
        nb.save(roi, os.path.join(out_dir, 'ROI_'+roi_name+'.nii.gz'))
        if qa:
            sl = xform_coord(roi.get_affine(), np.array(np.where(roi.get_data())).mean(axis=1)).round()
            outfile = os.path.join(out_dir, subcode+'_ROI_'+roi_name+'.png')
            show_brain(fa, sl=sl, overlay_file=roi, overlay_clip=[1,2], outfile=outfile)
        
def make_rois_fsl(subcode, roi_names, roi_dir='/data/ROIs/'):
    out_dir = os.path.join('/data/hcp/data',subcode)
    warp = os.path.join('/hcp',subcode,'MNINonLinear/xfms/standard2acpc_dc.nii.gz')
    ref = os.path.join('/hcp',subcode,'T1w', 'Diffusion', 'nodif_brain_mask.nii.gz')
    for i,roi_name in enumerate(roi_names):
        infile = os.path.join(roi_dir,roi_name)
        outfile = os.path.join(out_dir, 'ROI_'+roi_name+'_fsl.nii.gz')
        call(['applywarp', '-i', infile, '-r', ref, '-w', warp, '-o', outfile])