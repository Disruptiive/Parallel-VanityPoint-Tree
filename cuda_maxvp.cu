#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h> 
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <float.h>
#include <limits.h>

typedef struct {
    int *idx_elements;
    int size;
    int dimensions;
    int vp_idx;
    double median;
} vpNode_t;


void generatePoints(int N, int k,double *arr){
    int i;
    for(i=0;i<N*k;i++){
        arr[i] = ((float)rand()/(float)(RAND_MAX)) * 10.0;
    }
}

__device__
void swap(double *arr, int a, int b)
{
    double tmp = arr[a];
    arr[a] = arr[b];
    arr[b] = tmp;
}

__device__
int partition (double *arr, int left, int right)
{
    double pivot = arr[right]; // pivot
    int i = left,j;
    for (j = left; j <= (right - 1); j++)
    {
        // If current element is smaller than the pivot
        if (arr[j] < pivot)
        {
            swap(arr,i,j);
            i++;
        }
    }
    swap(arr,i,right);
    return i;
}

__device__
double quickselect(double *arr,int left,int right, int k){
    int index = partition(arr,left,right);
    if (index - left == k - 1)
        return arr[index];
    if (index - left > k - 1)
        return quickselect(arr, left, index - 1, k);
    return quickselect(arr, index + 1, right,k - index + left - 1);
}

//find median of an even array 
__device__
double findMedian(double *arr,int size){
    //find n/2 smallest and n/2+1 smallest number of the array sum them and divide by 2
    double median1 = quickselect(arr,0,size-1,size/2);
    double median2 = quickselect(arr,0,size-1,size/2+1);

    return (median1+median2)/2;
}

//calculate distance between 2 double multidimensional points
__device__
double calculateDistance(int p1_idx,int p2_idx, int dimensions, double *elements){
    int i,idx1,idx2;
    double result = 0;
    idx1 = p1_idx*dimensions;
    idx2 = p2_idx*dimensions;
    for (i=0;i<dimensions;i++){
        result += pow(elements[idx1]-elements[idx2],2);
        idx1++;
        idx2++;
    }
    result = sqrt(result);
    return result;
}
void initializeChildrenNodes(vpNode_t *node,vpNode_t *lower_child,vpNode_t *higher_child){
    for(int i=0;i<node->size;i++){
        lower_child[i].size = node->size/2;
        higher_child[i].size = node->size/2;
        lower_child[i].dimensions = node->dimensions;
        higher_child[i].dimensions = node->dimensions;

        cudaMalloc(&lower_child[i].idx_elements,lower_child->size*sizeof(int));
        cudaMalloc(&higher_child[i].idx_elements,higher_child->size*sizeof(int));
    }
}


__global__
void vanityPoint(vpNode_t *root,vpNode_t *lower, vpNode_t *higher,double *pts){
    extern __shared__ double distance_arr[];
    int i = blockIdx.x;
    for (int j = threadIdx.x; j < root[i].size; j += blockDim.x){
        distance_arr[j] = calculateDistance(j,root[i].vp_idx,root[i].dimensions,pts);
    }
    
    if(threadIdx.x==0){
        int counter_high = 0;
        int counter_low  = 0;
        root[i].median = findMedian(distance_arr,root[i].size);
        for(int j=0;j<root[i].size;j++){
            if(distance_arr[j]<root[i].median){
                lower[i].idx_elements[counter_low++] = i;
            }
            else{
                higher[i].idx_elements[counter_high++] = i;
            }
        }
        lower[i].vp_idx = lower[i].idx_elements[lower[i].size-1];
        higher[i].vp_idx = higher[i].idx_elements[higher[i].size-1];
    }

}
void vanityTree(vpNode_t *node,double *pts){
    vpNode_t *child_upper, *child_lower;
    cudaMallocManaged(&child_lower,sizeof(vpNode_t)*node->size);
    cudaMallocManaged(&child_upper,sizeof(vpNode_t)*node->size);
    initializeChildrenNodes(node,child_lower,child_upper);
    
    vanityPoint<<<node->size,node->size,node->size*sizeof(double)>>>(node,child_lower,child_upper,pts);
    cudaDeviceSynchronize();
    //printf("%d\n",node->size);
    
    if (node->size>2){
        vanityTree(child_lower,pts);
        vanityTree(child_upper,pts);
    }    
    /*
    for(int i =0; i<1024;i++){
        printf("%lf\n",node[i].median);
    }
    */
}

vpNode_t* initialiseRoot(int N, int d){
    vpNode_t *parent;
    cudaMallocManaged(&parent,sizeof(vpNode_t)*N);
    for(int i=0;i<N;i++){
        parent[i].size = N;
        parent[i].dimensions = d;
        cudaMallocManaged(&parent[i].idx_elements,sizeof(int)*parent[i].size);
        for(int j=0;j<parent[i].size;j++){
            parent[i].idx_elements[i] = j;
        }
        parent[i].vp_idx = i;
    }
    
    
    return parent;
}

int main(){
    int N,d;
    
    struct timeval t_start,t_end;
    double exec_time;
    
    N = 1024;
    d = 128;
    
    double *pts;
    cudaMallocManaged(&pts,N*d*(sizeof(double)));
    generatePoints(N,d,pts);

    vpNode_t *parent = initialiseRoot(N,d);

    gettimeofday(&t_start,NULL);

    vanityTree(parent,pts);

    gettimeofday(&t_end,NULL);
    exec_time = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
    exec_time += (t_end.tv_usec - t_start.tv_usec) / 1000.0;

    printf("Exec time: %.3lf s\n",exec_time/1000);
}