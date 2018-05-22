
# coding: utf-8

# # Initialization

# In[1]:


get_ipython().magic(u'matplotlib inline')
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
from scipy.spatial.distance import cdist

def warp_mni_to_img(target, warp, phy_coords):
    ''' Warp (physical-space) MNI-space coords to a target image. '''
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
    phy_coords[0,:] += xo
    phy_coords[1,:] += yo
    phy_coords[2,:] += zo
    if mni_ref==None:
        return phy_coords
    else:
        return xform_coords(np.linalg.inv(mni_ref.get_affine()), phy_coords)

#def xform_sl(subcode, sl_file):
#    out_dir = os.path.join('/data/hcp/data',subcode)
#    warp = nb.load(os.path.join('/hcp',subcode,'MNINonLinear/xfms/acpc_dc2standard.nii.gz'))
#    mni = nb.load('/usr/share/fsl/data/standard/MNI152_T1_1mm.nii.gz')
#    fa = nb.load(os.path.join('/hcp', subcode, 'T1w/T1w_acpc_dc_restore_1.25.nii.gz'))
#    sl = load_streamline_file(os.path.join(out_dir, sl_file))
#    slx = [warp_img_to_mni(fa, warp, np.array(s).T, mni).T.tolist() for s in sl]
#    return(slx)

def xform_sl(subcode, sl_file, img_coords=True):
    out_dir = os.path.join('/data/hcp/data',subcode)
    warp = nb.load(os.path.join('/hcp',subcode,'MNINonLinear/xfms/standard2acpc_dc.nii.gz'))
    mni = nb.load('/usr/share/fsl/data/standard/MNI152_T1_1mm.nii.gz')
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
    ref = nb.load(os.path.join(sub_dir,'T1w', 'Diffusion', 'nodif_brain_mask.nii.gz'))
    for i,roi_name in enumerate(roi_names):
        mni_roi = nb.load('/home/badamifs/scripts/rois' + roi_name + '.nii.gz')
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


# #Merge MNI-space track files to summarize group data

# In[2]:


from dipy.segment import select
from dipy_run import *
from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import ResampleFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric
feature = ResampleFeature(nb_points=50)
metric = AveragePointwiseEuclideanMetric(feature)
qb = QuickBundles(threshold=15., metric=metric)

def save_to_trackvis(streamlines, outname, dims, pixdim):
    hdr = nb.trackvis.empty_header()
    hdr['voxel_size'] = pixdim
    hdr['voxel_order'] = 'LAS'
    hdr['dim'] = dims
    trk = ((sl, None, None) for sl in streamlines)
    nb.trackvis.write(outname, trk, hdr, points_space='voxel')


# In[3]:


trk_file = 'Lamyg2LpMFG_optimized.trk'
subcodes = sorted([os.path.basename(d) for d in glob('/data/hcp/data/*') 
                   if os.path.exists(os.path.join(d,trk_file))])
len(subcodes)


# ##Get the centroid of the largest cluster for each subject

# In[4]:


# Load the fibers for each subject, transforming to MNI space 
slx = []
for sc in subcodes:
    slx.append(xform_sl(sc, trk_file))


# In[5]:


# cluster each subject's fibers
clusters = []
for sl in slx:
    clusters.append(qb.cluster([np.array(s) for s in sl]))


# In[6]:


# Find the largest cluster for each subject
slx_all = []
subject_ids = []
subject_id = 0
for c in clusters:
    if len(c)>0:
        #clust_num = np.array(map(len,c)).argmax()
        # Two biggest Clusters
        clust_num = np.array(map(len,c)).argsort()[-2:][::-1]
        slx_all.append(c.centroids[clust_num[0]])
        subject_ids.append(subject_id)
        if len(c) > 1:
            slx_all.append(c.centroids[clust_num[1]])
            subject_ids.append(subject_id)
    subject_id += 1


# In[8]:


# Save the centroid of the largest cluster for each subject
ni = nb.load('/usr/share/fsl/data/standard/MNI152_T1_1mm.nii.gz')
dims = ni.shape
pixdim = ni.header.get_zooms()
save_to_trackvis(slx_all, '/data/hcp/mni_all_centroids_'+trk_file, dims, pixdim)


# In[19]:


print 'Subjects whos major fibers pass through main area'

roi_center = np.array([64.0, 218-138.0, 85.0])
roi_radius = 20.0

distance = np.asarray([cdist(i, roi_center[np.newaxis, ...], 'euclidean') for i in slx_all]).squeeze()
min_distance_fiber_roi = np.min(distance, axis=1)
major_streams_1 = np.where(min_distance_fiber_roi < roi_radius)[0]

steams = [slx_all[i] for i in major_streams_1]
#save_to_trackvis(steams, '/data/hcp/major_centroids_'+trk_file, dims, pixdim)

subject_with_major_pathway_1 = np.unique(np.asarray(subject_ids)[major_streams_1])
print "%d subject have major pathway" % subject_with_major_pathway_1.shape[0]


# In[20]:


print 'Subjects whos second major fibers pass through main area'

roi_center = np.array([77.0, 218-98.0, 76.0])
roi_radius = 12.5

distance = np.asarray([cdist(i, roi_center[np.newaxis, ...], 'euclidean') for i in slx_all]).squeeze()
min_distance_fiber_roi = np.min(distance, axis=1)
major_streams_2 = np.where(min_distance_fiber_roi < roi_radius)[0]

steams = [slx_all[i] for i in major_streams_2]
#save_to_trackvis(steams, '/data/hcp/major_second_centroids_'+trk_file, dims, pixdim)

subject_with_major_pathway_2 = np.unique(np.asarray(subject_ids)[major_streams_2])
print "%d subject have major second pathway" % subject_with_major_pathway_2.shape[0]


# In[41]:


num_either = np.unique(np.hstack((subject_with_major_pathway_1,subject_with_major_pathway_2))).shape[0]
num_both = np.intersect1d(subject_with_major_pathway_1,subject_with_major_pathway_2).shape[0]
print('num subjects with either pathway: %d' % (num_either+1))
print('num subjects with both pathways: %d' % num_both)


# In[33]:





# In[ ]:


"""
print 'Find fiber that pass not through ROIs'

from dipy.tracking import utils
img = nb.load('/data/hcp/RVLPFC_15mm_54_27_12_1mm.nii.gz')
roi_data = img.get_data()
neighborhood_streamlines = utils.target(slx_all, roi_data, affine=np.eye(4), include=False)
neighborhood_streamlines = list(neighborhood_streamlines)

mapping_list = [True in (np.array_equal(x, y) for x in neighborhood_streamlines) for y in slx_all]
mapping_indices = np.where(np.asarray(mapping_list) == True)[0]

subjects = [list(subcodes[i] for i in mapping_indices)][0]

from scipy.spatial.distance import cdist

roi_coords = np.array(np.where(roi_data)).T
sl = [np.vstack((s[0,:],s[-1,:])) for s in neighborhood_streamlines]
Y = cdist(np.asarray(sl).reshape((-1, 3)), roi_coords, 'euclidean')
distance = np.min(Y, axis=1).reshape((-1, 2))
distance_to_roi = np.min(distance, axis=1)

d = {}
i = 0
for arg in subjects:
    d[arg] = distance_to_roi[i]
    i += 1
    
import operator
sorted_d = sorted(d.items(), key=operator.itemgetter(1), reverse=True)

print 'Subject and distance to ROI'
sorted_d
"""

