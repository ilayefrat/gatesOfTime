#define _GNU_SOURCE
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <x86intrin.h>
#include <time.h>
#include <unistd.h>
#include <string.h>

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
 // Ensure the write is complete before proceeding 
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
    int testCached = 0;
    int testUncached = 0;

    for (int i = 0; i < NUM_TRIALS; i++) {
        int should_cache = rand() % 2;
        //usleep(1500);
        uint64_t *in1 = tmp_mem();
        uint64_t *in2 = tmp_mem();
        uint64_t *out = tmp_mem();
        *in2 = 1;
        for (int i = 0; i < 500; i++) {
            not(in2, out);
        }
        
        if (!should_cache) {
            clear(out);
            set(in1);
            _mm_mfence(); 
            not(in1, out);
            if (!test(out)) success1++;
            testCached++; 
        } else {
            clear(out);
            clear(in1);
            not(in1, out);
            if (test(out)) success0++;
            testUncached++;   
        }
        free_tmp(in1);
        free_tmp(in2);
        free_tmp(out);
    }
    
    printf("NOT gate results:\n");
    printf("in cached   → out uncached: %.2f%% success\n", 100.0 * success1 / testCached);
    printf("in uncached → out cached: %.2f%% success\n", 100.0 * success0 / testUncached);

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
