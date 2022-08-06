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

struct vpNode{
    double **elements;
    int size;
    int dimensions;
    int vp_idx;
    double median;
    double **lower;
    double **upper;
};

void generatePoints(int N, int k,double **arr){
    int i,j;
    for(i=0;i<N;i++){
        for(j=0;j<k;j++){
            arr[i][j] = ((float)rand()/(float)(RAND_MAX)) * 10.0;
        }
    }
}

void swap(double *arr, int a, int b)
{
    double tmp = arr[a];
    arr[a] = arr[b];
    arr[b] = tmp;
}

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

double quickselect(double *arr,int left,int right, int k){
    int index = partition(arr,left,right);
    if (index - left == k - 1)
        return arr[index];
    if (index - left > k - 1)
        return quickselect(arr, left, index - 1, k);
    return quickselect(arr, index + 1, right,k - index + left - 1);
}

//find median of an even array 
double findMedian(double *arr,int size){
    //find n/2 smallest and n/2+1 smallest number of the array sum them and divide by 2
    double median1 = quickselect(arr,0,size-1,size/2);
    double median2 = quickselect(arr,0,size-1,size/2+1);
    return (median1+median2)/2;
}

//calculate distance between 2 double multidimensional points
double calculateDistance(double *point1, double *point2,int dimensions){
    int i;
    double result = 0;
    for (i=0;i<dimensions;i++){
        result += pow(point2[i]-point1[i],2);
    }
    result = sqrt(result);
    return result;  
}

void popuplateChildrenNodes(struct vpNode *node,struct vpNode *lower_child,struct vpNode *higher_child){
    lower_child->elements = node->lower;
    higher_child->elements = node->upper;
    lower_child->size = node->size/2;
    higher_child->size = node->size/2;
    lower_child->vp_idx = lower_child->size-1;
    higher_child->vp_idx = higher_child->size-1;
    lower_child->dimensions = node->dimensions;
    higher_child->dimensions = node->dimensions;

    lower_child->lower = malloc((lower_child->size/2)*sizeof(double*));
    lower_child->upper = malloc((lower_child->size/2)*sizeof(double*));
    higher_child->lower = malloc((higher_child->size/2)*sizeof(double*));
    higher_child->upper = malloc((higher_child->size/2)*sizeof(double*));
    
}

void vanityPoint(struct vpNode *node){
    int i,counter_low,counter_high;
    double *vp = node->elements[node->vp_idx];
    double *distance_arr = malloc(node->size*sizeof(double));

    for(i=0;i<node->size;i++){
        distance_arr[i] = calculateDistance(node->elements[i],node->elements[node->vp_idx],node->dimensions);
    }
    counter_low = 0;
    counter_high = 0;
    node->median = findMedian(distance_arr,node->size);

    for(i=0;i<node->size;i++){
        if(distance_arr[i]<node->median){
            #pragma omp critical (Inner)
            {
                node->lower[counter_low] = node->elements[i];
                counter_low++;
            }
        }
        else{
            #pragma omp critical (Outer)
            {
                node->upper[counter_high] = node->elements[i];
                counter_high++;
            }
        }
    }
    if (counter_high !=counter_low)
        printf("finished %d, %d\n",counter_low,counter_high);
    free(distance_arr);
}

void vanityTree(struct vpNode *node){
    struct vpNode *child_upper, *child_lower;
    vanityPoint(node);
    if (node->size>2){
        child_lower = malloc(sizeof(struct vpNode));
        child_upper = malloc(sizeof(struct vpNode));
        popuplateChildrenNodes(node,child_lower,child_upper);
        //free(node);
        vanityTree(child_lower);
        vanityTree(child_upper);
    }
    //else{
        //free(node);
    //}
}

int main(){
    int N,d,i;
    double **pts;
    struct timeval t_start,t_end;
    double exec_time;
    N = 1024;
    d = 128;
    

    pts = malloc(N*(sizeof(double*)));
    for(i=0;i<N;i++){
        pts[i] = malloc(d*sizeof(double));
    }

    generatePoints(N,d,pts);
    struct vpNode *parent = malloc(sizeof(struct vpNode)*N);
    for(int i=0;i<N;i++){
        parent[i].elements = pts;
        parent[i].size = N;
        parent[i].dimensions = d;
        parent[i].lower = malloc((parent->size/2)*sizeof(double*));
        parent[i].upper = malloc((parent->size/2)*sizeof(double*));
        parent[i].vp_idx = i;
    }
    gettimeofday(&t_start,NULL);
    
    for(int i=0;i<N;i++){

        vanityTree(&parent[i]);
    }

    gettimeofday(&t_end,NULL);
    exec_time = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
    exec_time += (t_end.tv_usec - t_start.tv_usec) / 1000.0;

    printf("Exec time: %.3lf s\n",exec_time/1000);
}