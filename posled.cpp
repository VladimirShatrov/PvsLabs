#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <numeric>

std::vector<float> generate_random_array(size_t size) {
   std::vector<float> array(size);
   std::random_device rd;
   std::mt19937 gen(rd());
   std::uniform_real_distribution<float> dist(0.0f, 1.0f);

   for (size_t i = 0; i < size; ++i) {
       array[i] = dist(gen);
   }
   return array;
}

float sequential_sum(const std::vector<float>& array) {
   return std::accumulate(array.begin(), array.end(), 0.0f);
}

int main() {
   const size_t array_size = 10000000;
   const int num_runs = 100;
   double total_time = 0.0;
   float reference_sum = 0.0f;

   auto array = generate_random_array(array_size);

   // Первый запуск (без учета в среднее)
   auto start = std::chrono::high_resolution_clock::now();
   reference_sum = sequential_sum(array);
   auto end = std::chrono::high_resolution_clock::now();
   std::chrono::duration<double> duration = end - start;

   std::cout << "Reference sum: " << reference_sum << std::endl;
   std::cout << "First run time: " << duration.count() << " seconds\n\n";

   // Основные запуски (100 раз)
   for (int i = 0; i < num_runs; ++i) {
       start = std::chrono::high_resolution_clock::now();
       float sum = sequential_sum(array);
       end = std::chrono::high_resolution_clock::now();
       duration = end - start;
       total_time += duration.count();

       // Проверка корректности
       if (std::abs(sum - reference_sum) > 1e-5) {
           std::cerr << "Error: sum mismatch! " << sum << " vs " << reference_sum << std::endl;
           return 1;
       }
   }

   // Вывод результатов
   double avg_time = total_time / num_runs;
   std::cout << "Average time over " << num_runs << " runs: " << avg_time << " seconds" << std::endl;
   std::cout << "Total time: " << total_time << " seconds" << std::endl;

   return 0;
}