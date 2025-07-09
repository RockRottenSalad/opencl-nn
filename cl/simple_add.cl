
void kernel simple_add(global const int* A, global const int* B, global const int* C, global int* D){ 
    D[get_global_id(0)]=A[get_global_id(0)]+B[get_global_id(0)]+C[get_global_id(0)];                 
} 

