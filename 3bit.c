#define _GNU_SOURCE
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <x86intrin.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include "test_cases.h"
#include "binary_arrays.h"

#define L1_CACHE_SIZE (48 * 1024)
#define NUM_TRIALS 1000
#define CACHE_HIT_THRESHOLD 190 
#define PAGE_SIZE 4096
#define CACHE_LINE_SIZE 64
#define NUM_VARS 3
#define ASSOCIATIVITY 12
#define TOTAL_VARS (NUM_TRIALS * NUM_VARS)
#define STRIDE (CACHE_LINE_SIZE * 2)

// Ramps up CPU to reduce initial variability
void warm_up_cpu() {
    for (volatile int i = 0; i < 10000000; ++i) {
        asm volatile("" ::: "memory");
    }
}

void clear(uint64_t *ptr) { 
    _mm_clflush(ptr);



}
void clear_conditional(uint64_t *ptr, int should_clear) {
    if (should_clear) {
        clear(ptr);
    }
}

int test(uint64_t *ptr) { 
    unsigned int junk;
    unsigned long long start, end;
    volatile int tmp;
    _mm_mfence();  // Ensure all previous memory operations are complete
    _mm_lfence();
    start = __rdtscp(&junk);  // serialize before read
    tmp = *ptr;
    _mm_lfence();  // Ensure the read is complete before measuring time
    end = __rdtscp(&junk);    // serialize after read

    unsigned long long delta = end - start;
    return delta < CACHE_HIT_THRESHOLD ? 1 : 0;
}

void set(uint64_t *ptr) { 
    *(volatile uint64_t *)ptr;
    //_mm_mfence();  // Ensure the write is complete before proceeding
}
void set_conditional(uint64_t *ptr, int should_set) {
    if (should_set) {
        set(ptr+should_set);
    }
}



void not(uint64_t *in, uint64_t *out) {
    if (*in == 0) {
        return;}
    for (int i = 0; i < 4; i++) {
        asm volatile(
            "imulq $1, %[out_addr], %[out_addr]"
            : [out_addr] "+r"(out)
            :
            : "memory"
        );
    }
    set(out);
    _mm_mfence();
}
void nand(uint64_t *in1, uint64_t *in2, uint64_t *out) {
    if (*in1 + *in2 == 0) {
        return;}
    for (int i = 0; i < 4; i++) {
        asm volatile(
            "imulq $1, %[out_addr], %[out_addr]"
            : [out_addr] "+r"(out)
            :
            : "memory"
        );
    }
    set(out); 
    _mm_mfence();
}


void nandMul(uint64_t *in1, uint64_t *in2, uint64_t *sum_out, uint64_t *carryout) {
    if (*in1 + *in2 == 0) {
        return;}
    for (int i = 0; i < 2; i++) {
        asm volatile(
            "imulq $1, %[out_addr], %[out_addr]"
            : [out_addr] "+r"(sum_out)
            :
            : "memory"
        );
    }
    for (int i = 0; i < 2; i++) {
        asm volatile(
            "imulq $1, %[out_addr], %[out_addr]"
            : [out_addr] "+r"(carryout)
            :
            : "memory"
        );
    }
    set(sum_out); 
    set(carryout);
    _mm_mfence();
}
void halfadder(uint64_t *in1, uint64_t *in2,uint64_t *in1copy, uint64_t *in2copy, uint64_t *tmp1, uint64_t *tmp2, uint64_t *tmp3,uint64_t *tmp4, uint64_t *sum_out) {
    nandMul(in1, in2, tmp1, tmp4);
    nand(in1copy, tmp1, tmp2);
    nand(in2copy, tmp4, tmp3);
    nand(tmp2, tmp3, sum_out);
}


void *tmp_mem() {
    static int global_cnt = 0;
    uint8_t *addr = aligned_alloc(PAGE_SIZE, PAGE_SIZE);
    memset(addr, 0, PAGE_SIZE);
    *addr = 0xdeadbeef + global_cnt;
    //addr += (rand() % (PAGE_SIZE - sizeof(uint64_t))); // still aligned to byte
    addr += 17*64;
    global_cnt += 1;
    return addr;
}

void free_tmp(void* addr) {
    addr = (void *) ((uint64_t) addr & ~(uint64_t)0xfff);
    free(addr);
}
void mod_pairs(const int in[6], int out[3]) {
    out[0] = (in[0] + in[3]) % 2;
    out[1] = (in[1] + in[4]) % 2;
    out[2] = (in[2] + in[5]) % 2;
}

void halfadder_all(uint64_t *in1, uint64_t *in2,uint64_t *in1copy, uint64_t *in2copy, uint64_t *tmp1, uint64_t *tmp2, uint64_t *tmp3,uint64_t *tmp4, uint64_t *sum_out, int should_cache1, int should_cache2, int all_zero, int res[3], int test_sum_out) {

    // clear_conditional(in1, !should_cache1);
    // clear_conditional(in2, !should_cache2);
    // clear_conditional(in1copy, !should_cache1);
    // clear_conditional(in2copy, !should_cache2);
    _mm_mfence();
    clear(in1);
    clear(in2);
    clear(in1copy);
    clear(in2copy);
    clear(sum_out);
    clear(tmp1);
    clear(tmp2);
    clear(tmp3);
    clear(tmp4);
    _mm_mfence();
    usleep(500);
    set_conditional(in1, should_cache1);
    set_conditional(in2, should_cache2);
    set_conditional(in1copy, should_cache1);
    set_conditional(in2copy, should_cache2);

    if(!all_zero) {
        _mm_mfence();
    }

    halfadder(in1, in2, in1copy, in2copy,  tmp1, tmp2, tmp3, tmp4, sum_out);
    res[test_sum_out] = test(sum_out);
}

int main() {
    warm_up_cpu();
    usleep(1500);

    int success0 = 0;
    int success1 = 0;
    int success2 = 0;
    int successTotal = 0;

    for (int i = 0; i < NUM_TRIALS; i++) {
        int should_cache[6];
        for (int j = 0; j < 6; j++) {
            should_cache[j] = binary_test_arrays[i][j];
        }
        int wanted_res[3];
        mod_pairs(should_cache, wanted_res);
        int all_zero = 1; 
        for (int j = 0; j < 6; j++) {
            if (should_cache[j] != 0) {
                all_zero = 0;
                break; // exit early, no need to check further
            }
        }

        int res[3];
        uint64_t *in1[3];
        uint64_t *in2[3];
        uint64_t *in3 = tmp_mem();
        uint64_t *in4 = tmp_mem();
        uint64_t *in1copy[3];
        uint64_t *in2copy[3];
        uint64_t *sum_out[3];
        uint64_t *tmp1[3];
        uint64_t *tmp2[3];
        uint64_t *tmp3[3];
        uint64_t *tmp4[3];
        *in3 = 1;
        *in4 = 1;
        for (int i = 0; i < 3; i++) {
            in1[i]      = tmp_mem();
            *in1[i] = 0; 
            in2[i]      = tmp_mem();
            *in2[i] = 0; 
            in1copy[i]  = tmp_mem();
            *in1copy[i] = 0;
            in2copy[i]  = tmp_mem();
            *in2copy[i] = 0;
            sum_out[i]  = tmp_mem();
            *sum_out[i] = 0;
            tmp1[i]     = tmp_mem();
            *tmp1[i] = 0;
            tmp2[i]     = tmp_mem();
            *tmp2[i] = 0;
            tmp3[i]     = tmp_mem();
            *tmp3[i] = 0;
            tmp4[i]     = tmp_mem();
            *tmp4[i] = 0;
            for (int j = 0; j < 500; j++) {
                nand(in3, in4, sum_out[i]);
                not(in3, sum_out[i]);
                nandMul(in3, in4, sum_out[i], sum_out[i]);
            }
            halfadder_all(in1[i], in2[i], in1copy[i], in2copy[i], tmp1[i], tmp2[i], tmp3[i], tmp4[i], sum_out[i], should_cache[i], should_cache[i + 3], all_zero, res, i);
            free_tmp(in1[i]);
            free_tmp(in2[i]);
            free_tmp(in1copy[i]);
            free_tmp(in2copy[i]);
            free_tmp(tmp1[i]);
            free_tmp(tmp2[i]);
            free_tmp(tmp3[i]);
            free_tmp(tmp4[i]);
            free_tmp(sum_out[i]);

        }
        free_tmp(in3);
        free_tmp(in4);


        

        if (res[0] == wanted_res[0]) {
            success0++;
        }
        if (res[1] == wanted_res[1]) {
            success1++;
        }
        if (res[2] == wanted_res[2]) {
            success2++;
        }
        if( res[0] == wanted_res[0] && res[1] == wanted_res[1] && res[2] == wanted_res[2]) {
            successTotal++;
        }


    }
    
    printf("3 bit gate results:\n");
    printf("success percentage, first bit: %.2f%% success\n", 100.0 * success0 / NUM_TRIALS);
    printf("success percentage, second bit: %.2f%% success\n", 100.0 * success1 / NUM_TRIALS);
    printf("success percentage, third bit: %.2f%% success\n", 100.0 * success2 / NUM_TRIALS);
    printf("success percentage, all bits: %.2f%% success\n", 100.0 * successTotal / NUM_TRIALS);

    return 0;
}


