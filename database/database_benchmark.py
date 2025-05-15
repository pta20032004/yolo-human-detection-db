"""
Database Performance Benchmark
Kiểm tra độ trễ tìm kiếm của cơ sở dữ liệu để đảm bảo yêu cầu < 15ms.
"""

import os
import sys
import time
import numpy as np
import statistics
import argparse
from concurrent.futures import ThreadPoolExecutor
import threading

# Thêm đường dẫn dự án
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.chroma_manager import ChromaFaceDB

class DatabaseBenchmark:
    """Đo lường hiệu suất hoạt động của cơ sở dữ liệu để kiểm tra độ trễ."""

    def __init__(self, db_path="./face_db/chroma_data"):
        """Khởi tạo benchmark."""
        print("Đang khởi tạo Database Benchmark...")
        self.face_db = ChromaFaceDB(db_path)
        self.lock = threading.Lock()
        print("Kết nối cơ sở dữ liệu thành công!")

    def _generate_test_embedding(self, dimension=128):
        """Tạo embedding ngẫu nhiên đã chuẩn hóa để kiểm tra."""
        embedding = np.random.randn(dimension).astype(np.float32)
        return embedding / np.linalg.norm(embedding)

    def _print_results(self, title, results, latency_check=False):
        """In kết quả benchmark một cách có cấu trúc."""
        print(f"\nKẾT QUẢ {title}:")
        if not results:
            print("Không có hoạt động thành công nào!")
            return

        print(f"Tỷ lệ thành công: {results.get('success_rate', 0)*100:.1f}%")
        print(f"Thời gian trung bình: {results.get('mean_time_ms', 0):.3f} ms")
        print(f"Thời gian trung vị: {results.get('median_time_ms', 0):.3f} ms")
        print(f"Thời gian P95: {results.get('p95_time_ms', 0):.3f} ms")
        print(f"Thời gian P99: {results.get('p99_time_ms', 0):.3f} ms")

        if "avg_results_per_search" in results:
            print(f"Số kết quả trung bình mỗi tìm kiếm: {results['avg_results_per_search']:.1f}")
        if "searches_per_second" in results:
             print(f"Số lượt tìm kiếm mỗi giây: {results['searches_per_second']:.1f}")
        if "errors" in results:
            print(f"Lỗi: {results['errors']}")


        if latency_check and 'mean_time_ms' in results and 'p95_time_ms' in results:
            print("\nKIỂM TRA YÊU CẦU ĐỘ TRỄ (< 15ms):")
            mean_time = results['mean_time_ms']
            p95_time = results['p95_time_ms']
            print(f"Trung bình: {mean_time:.3f}ms - {'ĐẠT' if mean_time < 15 else 'KHÔNG ĐẠT'}")
            print(f"P95: {p95_time:.3f}ms - {'ĐẠT' if p95_time < 15 else 'CHẬM'}")

    def benchmark_insertion(self, num_insertions=100, dimension=128):
        """Đo lường tốc độ chèn vector."""
        print(f"\nBENCHMARK CHÈN DỮ LIỆU ({num_insertions} vector, {dimension} chiều)")
        insertion_times = []
        successful_insertions = 0

        print("Warmup chèn dữ liệu...")
        for _ in range(10):
            self.face_db.add_face(self._generate_test_embedding(dimension), {"name": "test_warmup", "benchmark": True})

        print("Đang chạy benchmark chèn dữ liệu...")
        for i in range(num_insertions):
            embedding = self._generate_test_embedding(dimension)
            metadata = {"name": f"test_person_{i}", "benchmark": True, "batch_id": int(time.time())}
            start_time = time.perf_counter()
            face_id = self.face_db.add_face(embedding, metadata)
            end_time = time.perf_counter()

            if face_id:
                insertion_times.append((end_time - start_time) * 1000)
                successful_insertions += 1
            if (i + 1) % (num_insertions // 10 or 1) == 0:
                print(f"  Tiến độ: {i + 1}/{num_insertions}")

        if not insertion_times:
            self._print_results("CHÈN DỮ LIỆU", None)
            return None

        results = {
            "total_insertions": num_insertions,
            "successful_insertions": successful_insertions,
            "success_rate": successful_insertions / num_insertions if num_insertions > 0 else 0,
            "mean_time_ms": statistics.mean(insertion_times),
            "median_time_ms": statistics.median(insertion_times),
            "min_time_ms": min(insertion_times),
            "max_time_ms": max(insertion_times),
            "std_dev_ms": statistics.stdev(insertion_times) if len(insertion_times) > 1 else 0,
            "p95_time_ms": np.percentile(insertion_times, 95),
            "p99_time_ms": np.percentile(insertion_times, 99),
        }
        self._print_results("CHÈN DỮ LIỆU", results)
        return results

    def benchmark_search(self, num_searches=100, top_k=5, threshold=0.7, dimension=128):
        """Đo lường tốc độ tìm kiếm vector."""
        print(f"\nBENCHMARK TÌM KIẾM ({num_searches} lượt, Top K: {top_k}, Ngưỡng: {threshold}, {dimension} chiều)")

        db_info = self.face_db.get_database_info()
        if db_info.get('count', 0) < 50: # Đảm bảo có đủ dữ liệu
            print("Cơ sở dữ liệu có ít bản ghi. Đang thêm dữ liệu thử nghiệm...")
            for i in range(50 - db_info.get('count', 0)):
                self.face_db.add_face(self._generate_test_embedding(dimension), {"name": f"search_test_{i}", "benchmark": True})
            print("Đã thêm dữ liệu thử nghiệm.")

        search_times = []
        total_results_found = 0

        print("Warmup tìm kiếm...")
        for _ in range(10):
            self.face_db.search_faces(self._generate_test_embedding(dimension), top_k, threshold)

        print("Đang chạy benchmark tìm kiếm...")
        for i in range(num_searches):
            query_embedding = self._generate_test_embedding(dimension)
            start_time = time.perf_counter()
            results_found = self.face_db.search_faces(query_embedding, top_k, threshold)
            end_time = time.perf_counter()

            search_times.append((end_time - start_time) * 1000)
            total_results_found += len(results_found)
            if (i + 1) % (num_searches // 10 or 1) == 0:
                print(f"  Tiến độ: {i + 1}/{num_searches}")

        if not search_times:
            self._print_results("TÌM KIẾM", None, latency_check=True)
            return None

        results = {
            "total_searches": num_searches,
            "successful_searches": num_searches, # Giả định mọi tìm kiếm đều "thành công" về mặt thực thi
            "success_rate": 1.0,
            "mean_time_ms": statistics.mean(search_times),
            "median_time_ms": statistics.median(search_times),
            "min_time_ms": min(search_times),
            "max_time_ms": max(search_times),
            "std_dev_ms": statistics.stdev(search_times) if len(search_times) > 1 else 0,
            "p95_time_ms": np.percentile(search_times, 95),
            "p99_time_ms": np.percentile(search_times, 99),
            "avg_results_per_search": total_results_found / num_searches if num_searches > 0 else 0,
        }
        self._print_results("TÌM KIẾM", results, latency_check=True)
        return results

    def benchmark_concurrent_search(self, num_threads=5, searches_per_thread=20, dimension=128):
        """Đo lường hoạt động tìm kiếm đồng thời."""
        total_searches_planned = num_threads * searches_per_thread
        print(f"\nBENCHMARK TÌM KIẾM ĐỒNG THỜI ({num_threads} luồng, {searches_per_thread} lượt/luồng, Tổng: {total_searches_planned})")

        all_times = []
        successful_searches_count = 0
        errors_count = 0

        # Đảm bảo có đủ dữ liệu cho tìm kiếm đồng thời
        db_info = self.face_db.get_database_info()
        if db_info.get('count', 0) < 100: # Cần nhiều dữ liệu hơn cho kiểm tra đồng thời
            print("Cơ sở dữ liệu có ít bản ghi cho kiểm tra đồng thời. Đang thêm dữ liệu thử nghiệm...")
            for i in range(100 - db_info.get('count', 0)):
                self.face_db.add_face(self._generate_test_embedding(dimension), {"name": f"concurrent_search_test_{i}", "benchmark": True})
            print("Đã thêm dữ liệu thử nghiệm.")


        def search_worker():
            nonlocal successful_searches_count, errors_count
            thread_times = []
            for _ in range(searches_per_thread):
                try:
                    query_embedding = self._generate_test_embedding(dimension)
                    start_time = time.perf_counter()
                    self.face_db.search_faces(query_embedding, 5, 0.7)
                    end_time = time.perf_counter()
                    thread_times.append((end_time - start_time) * 1000)
                    with self.lock:
                        successful_searches_count += 1
                except Exception as e:
                    # print(f"Lỗi trong luồng: {e}") # Gỡ lỗi nếu cần
                    with self.lock:
                        errors_count += 1
            with self.lock:
                all_times.extend(thread_times)

        print("Đang chạy benchmark tìm kiếm đồng thời...")
        start_total_time = time.perf_counter()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(search_worker) for _ in range(num_threads)]
            for future in futures:
                future.result() # Chờ tất cả các luồng hoàn thành
        total_execution_time_sec = time.perf_counter() - start_total_time

        if not all_times:
            self._print_results("TÌM KIẾM ĐỒNG THỜI", {"errors": errors_count}, latency_check=True)
            return None

        results = {
            "total_searches_planned": total_searches_planned,
            "successful_searches": successful_searches_count,
            "errors": errors_count,
            "success_rate": successful_searches_count / total_searches_planned if total_searches_planned > 0 else 0,
            "total_time_seconds": total_execution_time_sec,
            "searches_per_second": successful_searches_count / total_execution_time_sec if total_execution_time_sec > 0 else 0,
            "mean_time_ms": statistics.mean(all_times),
            "median_time_ms": statistics.median(all_times),
            "p95_time_ms": np.percentile(all_times, 95),
            "p99_time_ms": np.percentile(all_times, 99),
            "std_dev_ms": statistics.stdev(all_times) if len(all_times) > 1 else 0,
        }
        self._print_results("TÌM KIẾM ĐỒNG THỜI", results, latency_check=True)
        return results

    def full_benchmark(self, num_insertions=100, num_searches=100, dimension=128):
        """Chạy bộ benchmark đầy đủ."""
        print("ĐANG CHẠY BENCHMARK CƠ SỞ DỮ LIỆU ĐẦY ĐỦ")
        print("="*50)

        db_info = self.face_db.get_database_info()
        print(f"Kích thước cơ sở dữ liệu ban đầu: {db_info.get('count', 0)} khuôn mặt")

        insertion_results = self.benchmark_insertion(num_insertions, dimension)
        search_results = self.benchmark_search(num_searches, dimension=dimension)
        concurrent_results = self.benchmark_concurrent_search(dimension=dimension)

        print("\nBÁO CÁO TỔNG KẾT")
        print("="*50)

        if insertion_results:
            print(f"Chèn dữ liệu: {insertion_results['mean_time_ms']:.3f}ms trung bình")
        if search_results:
            print(f"Tìm kiếm: {search_results['mean_time_ms']:.3f}ms trung bình")
            if insertion_results: # Chỉ in tổng hợp nếu cả hai đều chạy
                total_time_combined = insertion_results['mean_time_ms'] + search_results['mean_time_ms']
                print(f"Tổng hợp (chèn + tìm kiếm đơn lẻ): {total_time_combined:.3f}ms")
                print(f"TỔNG THỂ (dựa trên tìm kiếm đơn lẻ): {'ĐẠT' if search_results['mean_time_ms'] < 15 else 'KHÔNG ĐẠT'} (yêu cầu <15ms cho tìm kiếm)")

        if concurrent_results:
            print(f"Thông lượng tìm kiếm đồng thời: {concurrent_results['searches_per_second']:.1f} lượt tìm kiếm/giây")
            print(f"TỔNG THỂ (dựa trên tìm kiếm đồng thời P95): {'ĐẠT' if concurrent_results['p95_time_ms'] < 15 else 'KHÔNG ĐẠT'} (yêu cầu <15ms cho P95)")


        return {
            "insertion": insertion_results,
            "search": search_results,
            "concurrent": concurrent_results
        }

def main():
    parser = argparse.ArgumentParser(description='Benchmark Hiệu Suất Cơ Sở Dữ Liệu')
    parser.add_argument('--db_path', type=str, default='./face_db/chroma_data', help='Đường dẫn cơ sở dữ liệu')
    parser.add_argument('--insertions', type=int, default=100, help='Số lượt chèn để kiểm tra')
    parser.add_argument('--searches', type=int, default=100, help='Số lượt tìm kiếm để kiểm tra')
    parser.add_argument('--dimension', type=int, default=128, help='Chiều của embedding')
    parser.add_argument('--threads', type=int, default=5, help='Số luồng cho kiểm tra đồng thời')
    parser.add_argument('--searches_per_thread', type=int, default=20, help='Số lượt tìm kiếm mỗi luồng')
    parser.add_argument('--concurrent_only', action='store_true', help='Chỉ chạy kiểm tra tìm kiếm đồng thời')
    parser.add_argument('--insertion_only', action='store_true', help='Chỉ chạy benchmark chèn dữ liệu')
    parser.add_argument('--search_only', action='store_true', help='Chỉ chạy benchmark tìm kiếm')

    args = parser.parse_args()

    benchmark = DatabaseBenchmark(args.db_path)

    if args.insertion_only:
        benchmark.benchmark_insertion(args.insertions, args.dimension)
    elif args.search_only:
        benchmark.benchmark_search(args.searches, dimension=args.dimension)
    elif args.concurrent_only:
        benchmark.benchmark_concurrent_search(args.threads, args.searches_per_thread, args.dimension)
    else:
        benchmark.full_benchmark(args.insertions, args.searches, args.dimension)

if __name__ == "__main__":
    main()