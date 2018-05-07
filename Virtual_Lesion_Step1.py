
# coding: utf-8

# # Virtual Lesion
# This series of notebooks is used to determine the evidence for a pathway connecting two ROIs following those described in Pestilli et al.<sup>[1]</sup> and Leong et al.<sup>[2]</sup>.
# 
# Required steps before running this notebook:
# 
# <ol>
#     <li>Regions of interest (ROIs) were defined in MNI space and subsequently warped from MNI space to each individual subject’s native space</li>
#     <li>Constrained spherical deconvolution (CSD)<sup>[3]</sup> was used to estimate the fiber orientation distribution function (fODF). The fODFs were combined with tissue segmentations derived from the high resolution T1-weighted images to perform anatomically-constrained tractography<sup>[4]</sup></li>
#     <li>A whole-brain connectome was then generated by seeding all voxels in the white matter. These were filtered down using spherical-deconvolution informed filtering of tractograms (SIFT)<sup>[5]</sup> </li>
# </ol>
# 
# Calculation steps done in this notebook:
# 
# <ol>
#     <li>Generate fiber tracks seeding from ROI1 that pass through ROI2 and vise versa with MRtrix3</li>
#     <li>Remove the fibers from the previous step that do not start in one ROI and end in the other (The fibers that pass through the ROIs but do not terminate in them)</li>
#     <li>Combine all valid fiber tracks into one candiadate streamline set</li>
#     <li>Cluster the candiadate streamlines with dipys<sup>[6]</sup> QuickBundles algorithm<sup>[7]</sup> to remove outliers</li>
# </ol>
# 
# <sup>[1]</sup> <i>PMID: 25194848</i> <br/>
# <sup>[2]</sup> <i>PMID: 26748088</i> <br/>
# <sup>[3]</sup> <i>PMID: 18583153</i> <br/>
# <sup>[4]</sup> <i>PMID: 22705374</i> <br/>
# <sup>[5]</sup> <i>PMID: 23238430</i> <br/>
# <sup>[6]</sup> <i>PMID: 24600385</i> <br/>
# <sup>[7]</sup> <i>PMID: 23248578</i> <br/>

# In[3]:

import numpy as np
from nibabel import trackvis as tv
from dipy.segment.clustering import QuickBundles
import utilities
from dipy.segment.metric import ResampleFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric
import itertools
import MRTrix2TrackVis
import os


# In[11]:

# Library of Files
path = '/hcp/'
path_saveing = '/data/hcp/data/'

subjects = os.listdir(path_saveing)

subjects_sorted = sorted(subjects)
subjects_sorted.remove('.nii.gz')

for subject in subjects_sorted:
    print ('Process subject ' + subject)
    
    if os.path.isfile(os.path.join(path_saveing, subject, 'Lamyg2LpMFG_clustered2.trk')) == False:
        print "    Clustered File does not exist for this subject, start calculation."
    
        if os.path.isfile(os.path.join(path_saveing, subject, 'FOD.mif')) == True and os.path.isfile(os.path.join(path_saveing, subject, 'L_amyg_small_warped.nii.gz')) == True and os.path.isfile(os.path.join(path_saveing, subject, 'L_pMFG_warped.nii.gz')) == True:
            print "    All neccessary files there, continue ..."
    
            directory_output = os.path.join(path_saveing, subject)

            if os.path.isfile(os.path.join(path_saveing, subject, 'Lamyg2LpMFG_combined.tck')) == False:
                print '    Fiber Tracks do not exist, start First Fiber Fracking'
                cmd = "tckgen " + directory_output + "/FOD.mif " + directory_output + "Lamyg2LpMFG.tck -number 2500 -seed_image " + directory_output + "L_amyg_small_warped.nii.gz  -include " + directory_output + "L_pMFG_warped.nii.gz -force -maxnum 500000000 -act " + directory_output + "/5TT.mif -backtrack -crop_at_gmwmi -maxlength 250"
                os.system(cmd)

                print '    Start Second Fiber Tracking'
                cmd = "tckgen " + directory_output + "/FOD.mif " + directory_output + "Lamyg2LpMFG.tck -number 2500 -seed_image " + directory_output + "L_amyg_small_warped.nii.gz -include " + directory_output + "L_pMFG_warped.nii.gz -force -maxnum 500000000 -act " + directory_output + "/5TT.mif -backtrack -crop_at_gmwmi -maxlength 250"
                os.system(cmd)

                print '    First step to remove too long fiber from the first streamlines'
                cmd = "tckedit " + directory_output + "/Lamyg2LpMFG1.tck " + directory_output + "/Lamyg2LpMFG1_cut.tck -include " + directory_output + "/L_amyg_small_warped.nii.gz -test_ends_only -force"
                os.system(cmd)

                print '    Second step to remove too long fiber from the first streamlines'
                cmd = "tckedit " + directory_output + "/Lamyg2LpMFG1_cut.tck " + directory_output + "/Lamyg2LpMFG1_cut_cut.tck -include " + directory_output + "/L_pMFG_warped.nii.gz -test_ends_only -force"
                os.system(cmd)

                print '    First step to remove too long fiber from the second streamlines'
                cmd = "tckedit " + directory_output + "/Lamyg2LpMFG2.tck " + directory_output + "/Lamyg2LpMFG2_cut.tck -include " + directory_output + "/L_amyg_small_warped.nii.gz -test_ends_only -force"
                os.system(cmd)

                print '    Second step to remove too long fiber from the second streamlines'
                cmd = "tckedit " + directory_output + "/Lamyg2LpMFG2_cut.tck " + directory_output + "/Lamyg2LpMFG2_cut_cut.tck -include " + directory_output + "/L_pMFG_warped.nii.gz -test_ends_only -force"
                os.system(cmd)

                print '    Combine resulting streamlines'
                cmd = "tckedit " + directory_output + "/Lamyg2LpMFG1_cut_cut.tck " + directory_output + "/Lamyg2LpMFG2_cut_cut.tck " + directory_output + "/Lamyg2LpMFG_combined.tck  -force"
                os.system(cmd)
                
            else:
                f_in_nifti = os.path.join(path, subject, 'T1w/Diffusion/data.nii.gz')
                f_in_stream = os.path.join(directory_output, 'Lamyg2LpMFG_combined.tck')
                f_out_converted = os.path.join(directory_output, 'Lamyg2LpMFG_combined.trk')
                f_out_clustered = os.path.join(directory_output, 'Lamyg2LpMFG_clustered.trk')
                f_out_centroids = os.path.join(directory_output, 'Lamyg2LpMFG_centroids.trk')

                if os.path.isfile(f_in_nifti) == True:
                    print "    Can access raw nifti data, start conversion and clustering."

                    print '    Convert MRTrix streams to TrackVis'
                    try: 
                        MRTrix2TrackVis.convert_tck2trk(f_in_stream, f_in_nifti, f_out_converted)
                    except:
                        print 'Could not convert .tck to .trk'

                    print '    Cluster Steams'
                    try: 
                        streams, hdr = tv.read(f_out_converted)
                        streamlines = [i[0] for i in streams]

                        feature = ResampleFeature(nb_points=50)
                        metric = AveragePointwiseEuclideanMetric(feature=feature)
                        qb = QuickBundles(threshold=10., metric=metric)
                        clusters = qb.cluster(streamlines)

                        major_cluster = clusters > 60
                        major_path = []
                        for j in range(len(clusters)):
                            if major_cluster[j] == True:
                                major_path.append([streamlines[i] for i in clusters[j].indices])
                        major_streams = list(itertools.chain(*major_path))

                        strm = ((sl, None, None) for sl in major_streams)
                        tv.write(f_out_clustered, strm,  hdr_mapping=hdr)
                        
                        print '    All done'
                        
                    except:
                        print '    Could not Cluster streams'
                else:
                    print "    Could not load raw diffusion data, skip conversion and clustering."
        else:
            print "    Some input files are missing, skip this subject."
    else:
        print "    Clustered File exists already for this subject, skip calculation."


# In[ ]:



