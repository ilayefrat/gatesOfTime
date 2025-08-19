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
    int testCached = 0;
    int testUncached = 0;
    int testUncached1 = 0;
    int testUncached2 = 0;

    for (int i = 0; i < NUM_TRIALS; i++) {
        int should_cache = test_selection_array[i];
        //usleep(1500);
        uint64_t *in1 = tmp_mem();
        uint64_t *in2 = tmp_mem();
        uint64_t *in3 = tmp_mem();
        uint64_t *in4 = tmp_mem();
        uint64_t *in1copy = tmp_mem();
        uint64_t *in2copy = tmp_mem();
        uint64_t *in3copy = tmp_mem();
        uint64_t *in4copy = tmp_mem();
        uint64_t *sum_out = tmp_mem();
        uint64_t *carryout = tmp_mem();
        uint64_t *tmp1 = tmp_mem();
        uint64_t *tmp2 = tmp_mem();
        uint64_t *tmp3 = tmp_mem();
        uint64_t *tmp4 = tmp_mem();
        uint64_t *tmp5 = tmp_mem();
        *in1 = 0; 
        *in2 = 0;
        *in3 = 1;
        *in4 = 1;
        *in1copy = 0;
        *in2copy = 0;
        *in3copy = 0;
        *in4copy = 0;
        *tmp1 = 0;
        *tmp2 = 0;
        *tmp3 = 0;
        *tmp4 = 0;
        *tmp5 = 0;
        *sum_out = 0; 
        *carryout = 0;
        for (int i = 0; i < 500; i++) {
            nand(in3, in4, sum_out);
            not(in3, sum_out);
            nandMul(in3, in4, sum_out, carryout);
        }
        if (should_cache == 1) {
            clear(sum_out);
            clear(carryout);
            clear(tmp1);
            clear(tmp2);
            clear(tmp3);
            clear(tmp4);
            clear(tmp5);
            set(in1);
            set(in2);
            set(in1copy);
            set(in2copy);
            set(in3copy);
            set(in4copy);
            _mm_mfence();
            nandMul(in1, in2, tmp1, tmp5);
            nand(in1copy, tmp1, tmp2);
            nand(in2copy, tmp5, tmp3);
            nand(tmp2, tmp3, sum_out);
            int a = test(sum_out);
            nand(in3copy, in4copy, tmp4);
            not(tmp4, carryout);
            int b = test(carryout);
            if (!a && b) {success1++;}
            testCached++; 
        } else if (should_cache == 2) {
            clear(sum_out);
            clear(carryout);
            clear(tmp1);
            clear(tmp2);
            clear(tmp3);
            clear(tmp4);
            clear(tmp5);
            clear(in1);
            clear(in2);
            clear(in1copy);
            clear(in2copy);
            clear(in3copy);
            clear(in4copy);
            //xor
            nandMul(in1, in2, tmp1, tmp5);
            nand(in1copy, tmp1, tmp2);
            nand(in2copy, tmp5, tmp3);
            nand(tmp2, tmp3, sum_out);
            int a = test(sum_out);


            nand(in3copy, in4copy, tmp4);
            not(tmp4, carryout);
            int b = test(carryout);
            if (!a && !b) {success0++;}
            testUncached++;   
        } else if (should_cache == 3) {
            clear(sum_out);
            clear(carryout);
            clear(tmp1);
            clear(tmp2);
            clear(tmp3);
            clear(tmp4);
            clear(tmp5);
            clear(in1);
            clear(in1copy);
            clear(in3copy);
            set(in2);
            set(in2copy);
            set(in4copy);
            _mm_mfence();
            nandMul(in1, in2, tmp1, tmp5);
            nand(in1copy, tmp1, tmp2);
            nand(in2copy, tmp5, tmp3);
            nand(tmp2, tmp3, sum_out);
            int a = test(sum_out);
            nand(in3copy, in4copy, tmp4);
            not(tmp4, carryout);
            int b = test(carryout);
            if ((a && !b)) success2++;
            testUncached1++;
        } else {
            clear(sum_out);
            clear(carryout);
            clear(tmp1);
            clear(tmp2);
            clear(tmp3);
            clear(tmp4);
            clear(tmp5);
            clear(in2);
            clear(in2copy);
            clear(in4copy);
            set(in1);
            set(in1copy);
            set(in3copy);
            _mm_mfence();
            nandMul(in1, in2, tmp1, tmp5);
            nand(in1copy, tmp1, tmp2);
            nand(in2copy, tmp5, tmp3);
            nand(tmp2, tmp3, sum_out);
            int a = test(sum_out);
            nand(in3copy, in4copy, tmp4);
            not(tmp4, carryout);
            int b = test(carryout);
            if ((a && !b)) success3++;
            testUncached2++;
        }
        free_tmp(in1);
        free_tmp(in2);
        free_tmp(in3);
        free_tmp(in4);
        free_tmp(in1copy);
        free_tmp(in2copy);
        free_tmp(in3copy);
        free_tmp(in4copy);
        free_tmp(tmp1);
        free_tmp(tmp2);
        free_tmp(tmp3);
        free_tmp(tmp4);
        free_tmp(tmp5);
        free_tmp(sum_out);
        free_tmp(carryout);
    }
    
    printf("NAND gate results:\n");
    printf("in1, in2 cached   → out uncached: %.2f%% success\n", 100.0 * success1 / testCached);
    printf("in1, in2 uncached → out cached: %.2f%% success\n", 100.0 * success0 / testUncached);
    printf("in1 uncached, in2 cached → out cached: %.2f%% success\n", 100.0 * success2 / testUncached1);
    printf("in1 cached, in2 uncached → out cached: %.2f%% success\n", 100.0 * success3 / testUncached2);
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
