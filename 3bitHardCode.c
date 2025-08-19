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
#include "test_cases_1to7.h"

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
    _mm_sfence();
    usleep(1500);
}

//ad time afterrr

//what fence do i need in rdstcp

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

void clear_or_set(uint64_t *ptr, int should_clear) {
    if (should_clear) {
        clear(ptr);
    } else {
        set(ptr);
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
    uint8_t *addr = aligned_alloc(PAGE_SIZE, PAGE_SIZE);
    memset(addr, 0, PAGE_SIZE);
    addr += (rand() % (PAGE_SIZE - sizeof(uint64_t))); // still aligned to byte
    return addr;
}

void free_tmp(void* addr) {
    addr = (void *) ((uint64_t) addr & ~(uint64_t)0xfff);
    free(addr);
}

int main() {
    srand(time(NULL));
    warm_up_cpu();
    usleep(1500);

    int success0 = 0;
    int success1 = 0;
    int success2 = 0;
    int success3 = 0;
    int success4 = 0;
    int success5 = 0;
    int success6 = 0;
    int testCached = 0;
    int testUncached = 0;
    int testUncached1 = 0;
    int testUncached2 = 0;
    int testUncached3 = 0;
    int testUncached4 = 0;
    int testUncached5 = 0;

    for (int i = 0; i < NUM_TRIALS; i++) {
        int should_cache = test_selection_array_1to7[i];
        int res[3];
        //usleep(1500);
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
        }
        for (int i = 0; i < 500; i++) {
            nand(in3, in4, sum_out[0]);
            not(in3, sum_out[0]);
            nandMul(in3, in4, sum_out[0], sum_out[1]);
        }
        if (should_cache == 1) {
            for (int i = 0; i < 3; i++) {
                clear(sum_out[i]);
                clear(tmp1[i]);
                clear(tmp2[i]);
                clear(tmp3[i]);
                clear(tmp4[i]);
                set(in1[i]);
                set(in2[i]);
                set(in1copy[i]);
                set(in2copy[i]);
                _mm_mfence();
                halfadder(in1[i], in2[i], in1copy[i], in2copy[i],  tmp1[i], tmp2[i], tmp3[i], tmp4[i], sum_out[i]);
                res[i] = test(sum_out[i]);
            } 
            if (!res[0] && !res[1] && !res[2]) {success1++;}
            testCached++; 
        } else if (should_cache == 2) {
            for (int i = 0; i < 3; i++) {
                clear(in1[i]);
                clear(in1copy[i]);
                clear(in2[i]);
                clear(in1copy[i]);
                clear(sum_out[i]);
                clear(tmp1[i]);
                clear(tmp2[i]);
                clear(tmp3[i]);
                clear(tmp4[i]);
                halfadder(in1[i], in2[i], in1copy[i], in2copy[i],  tmp1[i], tmp2[i], tmp3[i], tmp4[i], sum_out[i]);
                res[i] = test(sum_out[i]);
            } 
            if (!res[0] && !res[1] && !res[2]) {success0++;}
            testUncached++; 

        } else if (should_cache == 3) {
            for (int i = 0; i < 3; i++) {
                clear(in1[i]);
                clear(in1copy[i]);
                clear(sum_out[i]);
                clear(tmp1[i]);
                clear(tmp2[i]);
                clear(tmp3[i]);
                clear(tmp4[i]);
                set(in2[i]);
                set(in2copy[i]);
                _mm_mfence();
                halfadder(in1[i], in2[i], in1copy[i], in2copy[i],  tmp1[i], tmp2[i], tmp3[i], tmp4[i], sum_out[i]);
                res[i] = test(sum_out[i]);
            }   
            if (res[0] && res[1] && res[2]) {success2++;}
            testUncached1++; 
        } else if (should_cache == 4){
            for (int i = 0; i < 3; i++) {
                clear(in2[i]);
                clear(in2copy[i]);
                clear(sum_out[i]);
                clear(tmp1[i]);
                clear(tmp2[i]);
                clear(tmp3[i]);
                clear(tmp4[i]);
                set(in1[i]);
                set(in1copy[i]);
                _mm_mfence();
                halfadder(in1[i], in2[i], in1copy[i], in2copy[i],  tmp1[i], tmp2[i], tmp3[i], tmp4[i], sum_out[i]);
                res[i] = test(sum_out[i]);
            }   
            if (res[0] && res[1] && res[2]) {success3++;}
            testUncached2++; 
        }
        else if (should_cache == 5){
            clear(sum_out[0]);
            clear(tmp1[0]);
            clear(tmp2[0]);
            clear(tmp3[0]);
            clear(tmp4[0]);
            set(in1[0]);    
            set(in2[0]);
            set(in1copy[0]);
            set(in2copy[0]);
            _mm_mfence();
            halfadder(in1[0], in2[0], in1copy[0], in2copy[0],  tmp1[0], tmp2[0], tmp3[0], tmp4[0], sum_out[0]);
            res[0] = test(sum_out[0]);

            clear(in1[1]);
            clear(in1copy[1]);
            clear(sum_out[1]);
            clear(tmp1[1]);
            clear(tmp2[1]);
            clear(tmp3[1]);
            clear(tmp4[1]);
            set(in2[1]);
            set(in2copy[1]);
            _mm_mfence();
            halfadder(in1[1], in2[1], in1copy[1], in2copy[1],  tmp1[1], tmp2[1], tmp3[1], tmp4[1], sum_out[1]);
            res[1] = test(sum_out[1]);

            clear(in1[2]);
            clear(in1copy[2]);
            clear(sum_out[2]);
            clear(tmp1[2]);
            clear(tmp2[2]);
            clear(tmp3[2]);
            clear(tmp4[2]);
            set(in2[2]);
            set(in2copy[2]);
            _mm_mfence();
            halfadder(in1[2], in2[2], in1copy[2], in2copy[2],  tmp1[2], tmp2[2], tmp3[2], tmp4[2], sum_out[2]);
            res[2] = test(sum_out[2]);
              
            if (!res[0] && res[1] && res[2]) {success4++;}
            testUncached3++; 
        }
        else if (should_cache == 6){
            clear(sum_out[0]);
            clear(tmp1[0]);
            clear(tmp2[0]);
            clear(tmp3[0]);
            clear(tmp4[0]);
            set(in1[0]);    
            set(in2[0]);
            set(in1copy[0]);
            set(in2copy[0]);
            _mm_mfence();
            halfadder(in1[0], in2[0], in1copy[0], in2copy[0],  tmp1[0], tmp2[0], tmp3[0], tmp4[0], sum_out[0]);
            res[0] = test(sum_out[0]);

            clear(sum_out[1]);
            clear(tmp1[1]);
            clear(tmp2[1]);
            clear(tmp3[1]);
            clear(tmp4[1]);
            set(in1[1]);
            set(in2[1]);
            set(in1copy[1]);
            set(in2copy[1]);
            _mm_mfence();
            halfadder(in1[1], in2[1], in1copy[1], in2copy[1],  tmp1[1], tmp2[1], tmp3[1], tmp4[1], sum_out[1]);
            res[1] = test(sum_out[1]);

            clear(in1[2]);
            clear(in1copy[2]);
            clear(sum_out[2]);
            clear(tmp1[2]);
            clear(tmp2[2]);
            clear(tmp3[2]);
            clear(tmp4[2]);
            set(in2[2]);
            set(in2copy[2]);
            _mm_mfence();
            halfadder(in1[2], in2[2], in1copy[2], in2copy[2],  tmp1[2], tmp2[2], tmp3[2], tmp4[2], sum_out[2]);
            res[2] = test(sum_out[2]);
              
            if (!res[0] && !res[1] && res[2]) {success5++;}
            testUncached4++; 
        }
        else{
            clear(in2[0]);
            clear(in2copy[0]);
            clear(sum_out[0]);
            clear(tmp1[0]);
            clear(tmp2[0]);
            clear(tmp3[0]);
            clear(tmp4[0]);
            set(in1[0]);    
            set(in1copy[0]);
            _mm_mfence();
            halfadder(in1[0], in2[0], in1copy[0], in2copy[0],  tmp1[0], tmp2[0], tmp3[0], tmp4[0], sum_out[0]);
            res[0] = test(sum_out[0]);

            clear(sum_out[1]);
            clear(tmp1[1]);
            clear(tmp2[1]);
            clear(tmp3[1]);
            clear(tmp4[1]);
            set(in1[1]);
            set(in2[1]);
            set(in1copy[1]);
            set(in2copy[1]);
            _mm_mfence();
            halfadder(in1[1], in2[1], in1copy[1], in2copy[1],  tmp1[1], tmp2[1], tmp3[1], tmp4[1], sum_out[1]);
            res[1] = test(sum_out[1]);

            clear(in1[2]);
            clear(in1copy[2]);
            clear(sum_out[2]);
            clear(tmp1[2]);
            clear(tmp2[2]);
            clear(tmp3[2]);
            clear(tmp4[2]);
            set(in2[2]);
            set(in2copy[2]);
            _mm_mfence();
            halfadder(in1[2], in2[2], in1copy[2], in2copy[2],  tmp1[2], tmp2[2], tmp3[2], tmp4[2], sum_out[2]);
            res[2] = test(sum_out[2]);
              
            if (res[0] && !res[1] && res[2]) {success6++;}
            testUncached5++; 
        }
        for (int i = 0; i < 3; i++) {
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
    }
    
    printf("3 bit gate results:\n");
    printf("in1[0-2], in2[0-2] cached   → out[0-2] uncached: %.2f%% success\n", 100.0 * success1 / testCached);
    printf("in1, in2 uncached → out[0-2] uncached: %.2f%% success\n", 100.0 * success0 / testUncached);
    printf("in1[0-2] uncached, in2[0-2] cached → out[0-2] cached: %.2f%% success\n", 100.0 * success2 / testUncached1);
    printf("in1 cached[0-2], in2[0-2] uncached → out[0-2] cached: %.2f%% success\n", 100.0 * success3 / testUncached2);
    printf("in1[0] cached, in1[1-2] uncached, in2[0-2] cached → out[0] cached, out[1-2] uncached: %.2f%% success\n", 100.0 * success4 / testUncached3);
    printf("in1[0-1] cached, in1[2] uncached, in2[0-2] cached → out[0-1] cached, out[2] uncached: %.2f%% success\n", 100.0 * success5 / testUncached4);
    printf("in1[0-1] cached, in1[2] uncached,, in2[0] uncached, in2[1-2] cached → out[0-2] cached: %.2f%% success\n", 100.0 * success6 / testUncached5);
    return 0;
}




//gcc -O2 -o allocnot allocnot.c
//printf("%d\n", test(in));
//printf("%d\n", test(out));
// printf("%llu\n", *in); 
//printf("%p\n", (void*)in);
//printf("%p\n", (void*)out);

        // print_cache_info("in1", in1);
        // print_cache_info("in2", in2);
        // print_cache_info("out", out);
