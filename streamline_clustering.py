import numpy as np
from nibabel import trackvis as tv
from dipy.segment.clustering import QuickBundles
import utilities
from dipy.segment.metric import ResampleFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric
import itertools
import MRTrix2TrackVis
import os

def clustering():
	parser = argparse.ArgumentParser(prog='streamline_clustering.py',
                                     description=__doc__)
	parser.add_argument('-i', '--input', type=str, required=True,
                        help="Raw data directory")
	parser.add_argument('-o', '--output', type=str, required=True,
                        help="Processed data directory")
	args = parser.parse_args()

	input_dir = op.abspath(args.input)
	directory_output = op.abspath(args.output)

	f_in_nifti = os.path.join(input_dir, 'T1w/Diffusion/data.nii.gz')
	f_in_stream = os.path.join(directory_output, 'Lamyg2LpMFG_combined.tck')
	f_out_converted = os.path.join(directory_output, 'Lamyg2LpMFG_combined.trk')
	f_out_clustered = os.path.join(directory_output, 'Lamyg2LpMFG_clustered.trk')
	f_out_centroids = os.path.join(directory_output, 'Lamyg2LpMFG_centroids.trk')

	#MRTrix2TrackVis.convert_tck2trk(f_in_stream, f_in_nifti, f_out_converted)
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
		print 'All done'
if __name__ == "__main__":
    clustering()