from os import listdir
import numpy as np 
import argparse
from inception import find_relevance, find_relevance_array, pairwise_similarities
from saliency import get_saliency_per_grid
from metrics import err_trajectory,gumbel_max_sample_array,rbp_trajectory

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--list', default='all', choices=['servers', 'storage', 'all'],) 

    parser.add_argument('-image_dir', type=str, default='generated_images',  help='directory that includes all the generated images')
    parser.add_argument('-target_image', type=str, default='pandatarget.png' ,  help='path to the target image')
    parser.add_argument('-metric', type=str, default='rbp',  choices=['rbp', 'err'],help='choice of base evaluation metric')
    parser.add_argument('-trajectory', type=str, default='saliency',  choices=['saliency', 'order'],help='choice of scanning trajectories in the grid')
    parser.add_argument('-gamma', type=float, default=0.8,help='persistency parameter')
    parser.add_argument('-n_samples', type=int, default=50,help='number of sampled trajectories')
    parser.add_argument('-variety', type=bool, default=True,help='Considering diversity of generated images or not')

    args = parser.parse_args()

    saliency_pred = np.array(get_saliency_per_grid(args.image_dir))
    images = [ args.image_dir + "/"+i for i in sorted(listdir(args.image_dir)) ]
    if args.variety==True:
        cosine_matrix=pairwise_similarities(images)

    original_relevance=find_relevance_array(args.target_image,images)
    print('saliency',saliency_pred)

    total_eval=[]
    for n in range(int(args.n_samples)):
        if args.trajectory == 'saliency':
            path=(gumbel_max_sample_array(saliency_pred))
        else:
            path = list(range(len(images)))

        if args.variety==False: 
            relevance = original_relevance
        else:
            relevance=original_relevance[:] 
            for i in range(len(path)):
                max_sim=0
                for j in range(i):
                    if cosine_matrix[path[i],path[j]] > max_sim:
                        max_sim=cosine_matrix[path[i],path[j]]
                
                relevance[path[i]] = relevance[path[i]] * (1 - max_sim)

        if args.metric=='rbp':
            total_eval.append(rbp_trajectory(relevance,path,args.gamma))
        elif args.metric =='err':
            total_eval.append(err_trajectory(relevance,path,args.gamma))

    print('The quality of the gird of generated images in '+args.image_dir+' directory is evaluated as :' )
    print('metric', args.metric)
    print('variety', args.variety)
    if args.variety==True: 
        print('trajectory', args.trajectory)
    print('evaluation:', str(np.mean(total_eval)))
