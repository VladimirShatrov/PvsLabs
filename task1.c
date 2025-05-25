#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define MIN_SIZE 100000  // Минимальный размер массива
#define MAX_SIZE 10000000 // Максимальный размер массива
#define NUM_ITERATIONS 100 // Количество итераций для усреднения времени

// Функция для заполнения массива случайными числами
void fill_array_random(int *arr, int size) {
    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % 100; // Случайные числа от 0 до 99
    }
}

// Последовательная сумма элементов массива
long long sequential_sum(int *arr, int size) {
    long long sum = 0;
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }
    return sum;
}

// Параллельная сумма элементов массива
long long parallel_sum(int *arr, int size, int num_threads) {
    long long sum = 0;
    omp_set_num_threads(num_threads);
#pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }
    return sum;
}

int main() {
    int size, num_threads;
    int *arr;
    clock_t start_time, end_time;
    double sequential_time, parallel_time;
    long long sequential_result, parallel_result;
    double total_sequential_time, total_parallel_time;

    // Чтение размера массива из переменной окружения SIZE
    char *size_str = getenv("SIZE");
    if (size_str == NULL) {
        fprintf(stderr, "Ошибка: не задана переменная окружения SIZE\n");
        return 1;
    }
    size = atoi(size_str);

    // Проверка размера массива
    if (size < MIN_SIZE || size > MAX_SIZE) {
        fprintf(stderr, "Ошибка: размер массива должен быть между %d и %d\n", MIN_SIZE, MAX_SIZE);
        return 1;
    }

    // Выделяем память под массив
    arr = (int *)malloc(size * sizeof(int));
    if (arr == NULL) {
        fprintf(stderr, "Ошибка выделения памяти.\n");
        return 1;
    }

    // Заполняем массив случайными числами
    fill_array_random(arr, size);

    // --- Последовательный вариант ---
    printf("--- Последовательный вариант ---\n");
    total_sequential_time = 0.0;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        start_time = clock();
        sequential_result = sequential_sum(arr, size);
        end_time = clock();
        sequential_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
        total_sequential_time += sequential_time;
    }
    printf("Среднее время последовательной суммы: %f секунд\n", total_sequential_time / NUM_ITERATIONS);
    printf("Сумма: %lld\n", sequential_result);

    // --- Параллельный вариант ---
    printf("--- Параллельный вариант ---\n");

    // Чтение количества потоков из переменной окружения NUM_THREADS
    char *num_threads_str = getenv("NUM_THREADS");
    if (num_threads_str == NULL) {
        fprintf(stderr, "Ошибка: не задана переменная окружения NUM_THREADS\n");
        return 1;
    }
    num_threads = atoi(num_threads_str);

    // Проверка количества потоков
    if (num_threads <= 0) {
        fprintf(stderr, "Ошибка: количество потоков должно быть положительным.\n");
        return 1;
    }
    total_parallel_time = 0.0;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        start_time = clock();
        parallel_result = parallel_sum(arr, size, num_threads);
        end_time = clock();
        parallel_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
        total_parallel_time += parallel_time;
    }
    printf("Среднее время параллельной суммы (с %d потоками): %f секунд\n", num_threads, total_parallel_time / NUM_ITERATIONS);
    printf("Сумма: %lld\n", parallel_result);

    // Освобождаем память
    free(arr);

    return 0;
}