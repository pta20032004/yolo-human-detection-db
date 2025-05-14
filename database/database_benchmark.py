"""
Database Performance Benchmark
Test database search latency để ensure < 15ms requirement
"""

import os
import sys
import time
import numpy as np
import statistics
import argparse
from concurrent.futures import ThreadPoolExecutor
import threading

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.chroma_manager import ChromaFaceDB

class DatabaseBenchmark:
    """Benchmark database operations for latency testing"""
    
    def __init__(self, db_path="./face_db/chroma_data"):
        """Initialize benchmark"""
        print("🚀 Initializing Database Benchmark...")
        self.face_db = ChromaFaceDB(db_path)
        self.lock = threading.Lock()
        print("✅ Database connection established!")
    
    def generate_test_embedding(self, dimension=128):
        """Generate random normalized embedding for testing"""
        # Create random vector
        embedding = np.random.randn(dimension).astype(np.float32)
        # L2 normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    
    def benchmark_insertion(self, num_insertions=100, dimension=128):
        """
        Benchmark vector insertion speed
        
        Args:
            num_insertions (int): Number of vectors to insert
            dimension (int): Embedding dimension
        
        Returns:
            dict: Insertion benchmark results
        """
        print(f"\n📥 INSERTION BENCHMARK")
        print(f"Vectors to insert: {num_insertions}")
        print(f"Dimension: {dimension}")
        
        insertion_times = []
        successful_insertions = 0
        
        # Warmup
        print("Warming up...")
        for _ in range(10):
            embedding = self.generate_test_embedding(dimension)
            metadata = {"name": f"test_warmup", "benchmark": True}
            self.face_db.add_face(embedding, metadata)
        
        print("Running insertion benchmark...")
        
        # Actual benchmark
        for i in range(num_insertions):
            # Generate test embedding
            embedding = self.generate_test_embedding(dimension)
            metadata = {
                "name": f"test_person_{i}",
                "benchmark": True,
                "batch_id": int(time.time())
            }
            
            # Measure insertion time
            start_time = time.perf_counter()
            face_id = self.face_db.add_face(embedding, metadata)
            end_time = time.perf_counter()
            
            if face_id:
                insertion_time = (end_time - start_time) * 1000  # Convert to ms
                insertion_times.append(insertion_time)
                successful_insertions += 1
            
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{num_insertions}")
        
        # Calculate statistics
        if insertion_times:
            results = {
                "total_insertions": num_insertions,
                "successful_insertions": successful_insertions,
                "success_rate": successful_insertions / num_insertions,
                "mean_time_ms": statistics.mean(insertion_times),
                "median_time_ms": statistics.median(insertion_times),
                "min_time_ms": min(insertion_times),
                "max_time_ms": max(insertion_times),
                "std_dev_ms": statistics.stdev(insertion_times) if len(insertion_times) > 1 else 0,
                "p95_time_ms": np.percentile(insertion_times, 95),
                "p99_time_ms": np.percentile(insertion_times, 99),
                "all_times": insertion_times
            }
            
            print(f"\n📊 INSERTION RESULTS:")
            print(f"Success rate: {results['success_rate']*100:.1f}%")
            print(f"Mean time: {results['mean_time_ms']:.3f} ms")
            print(f"Median time: {results['median_time_ms']:.3f} ms")
            print(f"P95 time: {results['p95_time_ms']:.3f} ms")
            print(f"P99 time: {results['p99_time_ms']:.3f} ms")
            
            return results
        else:
            print("❌ No successful insertions!")
            return None
    
    def benchmark_search(self, num_searches=100, top_k=5, threshold=0.7, dimension=128):
        """
        Benchmark vector search speed
        
        Args:
            num_searches (int): Number of search operations
            top_k (int): Number of results to return
            threshold (float): Similarity threshold
            dimension (int): Embedding dimension
        
        Returns:
            dict: Search benchmark results
        """
        print(f"\n🔍 SEARCH BENCHMARK")
        print(f"Searches to perform: {num_searches}")
        print(f"Top K: {top_k}")
        print(f"Threshold: {threshold}")
        print(f"Dimension: {dimension}")
        
        # Ensure we have some data in database
        db_info = self.face_db.get_database_info()
        if db_info.get('count', 0) < 10:
            print("⚠️  Database has few records. Adding test data...")
            for i in range(50):
                embedding = self.generate_test_embedding(dimension)
                metadata = {"name": f"search_test_{i}", "benchmark": True}
                self.face_db.add_face(embedding, metadata)
            print("✅ Test data added")
        
        search_times = []
        successful_searches = 0
        total_results_found = 0
        
        # Warmup
        print("Warming up...")
        for _ in range(10):
            query_embedding = self.generate_test_embedding(dimension)
            results = self.face_db.search_faces(query_embedding, top_k, threshold)
        
        print("Running search benchmark...")
        
        # Actual benchmark
        for i in range(num_searches):
            # Generate query embedding
            query_embedding = self.generate_test_embedding(dimension)
            
            # Measure search time
            start_time = time.perf_counter()
            results = self.face_db.search_faces(query_embedding, top_k, threshold)
            end_time = time.perf_counter()
            
            search_time = (end_time - start_time) * 1000  # Convert to ms
            search_times.append(search_time)
            successful_searches += 1
            total_results_found += len(results)
            
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{num_searches}")
        
        # Calculate statistics
        if search_times:
            results = {
                "total_searches": num_searches,
                "successful_searches": successful_searches,
                "success_rate": successful_searches / num_searches,
                "mean_time_ms": statistics.mean(search_times),
                "median_time_ms": statistics.median(search_times),
                "min_time_ms": min(search_times),
                "max_time_ms": max(search_times),
                "std_dev_ms": statistics.stdev(search_times) if len(search_times) > 1 else 0,
                "p95_time_ms": np.percentile(search_times, 95),
                "p99_time_ms": np.percentile(search_times, 99),
                "avg_results_per_search": total_results_found / successful_searches,
                "all_times": search_times
            }
            
            print(f"\n📊 SEARCH RESULTS:")
            print(f"Success rate: {results['success_rate']*100:.1f}%")
            print(f"Mean time: {results['mean_time_ms']:.3f} ms")
            print(f"Median time: {results['median_time_ms']:.3f} ms")
            print(f"P95 time: {results['p95_time_ms']:.3f} ms")
            print(f"P99 time: {results['p99_time_ms']:.3f} ms")
            print(f"Avg results per search: {results['avg_results_per_search']:.1f}")
            
            # Check 15ms requirement
            print(f"\n🎯 REQUIREMENT CHECK:")
            if results['mean_time_ms'] < 15:
                print(f"✅ PASSED: {results['mean_time_ms']:.3f}ms < 15ms target")
            else:
                print(f"❌ FAILED: {results['mean_time_ms']:.3f}ms > 15ms target")
            
            if results['p95_time_ms'] < 15:
                print(f"✅ P95 PASSED: {results['p95_time_ms']:.3f}ms < 15ms")
            else:
                print(f"⚠️  P95 SLOW: {results['p95_time_ms']:.3f}ms > 15ms")
            
            return results
        else:
            print("❌ No successful searches!")
            return None
    
    def benchmark_concurrent_search(self, num_threads=5, searches_per_thread=20, dimension=128):
        """
        Benchmark concurrent search operations
        
        Args:
            num_threads (int): Number of concurrent threads
            searches_per_thread (int): Searches per thread
            dimension (int): Embedding dimension
        
        Returns:
            dict: Concurrent search results
        """
        print(f"\n🔄 CONCURRENT SEARCH BENCHMARK")
        print(f"Threads: {num_threads}")
        print(f"Searches per thread: {searches_per_thread}")
        print(f"Total searches: {num_threads * searches_per_thread}")
        
        all_times = []
        successful_searches = 0
        errors = 0
        
        def search_worker(thread_id, num_searches):
            """Worker function for concurrent searches"""
            nonlocal successful_searches, errors
            thread_times = []
            
            for i in range(num_searches):
                try:
                    query_embedding = self.generate_test_embedding(dimension)
                    
                    start_time = time.perf_counter()
                    results = self.face_db.search_faces(query_embedding, 5, 0.7)
                    end_time = time.perf_counter()
                    
                    search_time = (end_time - start_time) * 1000
                    thread_times.append(search_time)
                    
                    with self.lock:
                        successful_searches += 1
                except Exception as e:
                    with self.lock:
                        errors += 1
                    print(f"Error in thread {thread_id}: {e}")
            
            with self.lock:
                all_times.extend(thread_times)
        
        # Run concurrent searches
        start_total = time.time()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for i in range(num_threads):
                future = executor.submit(search_worker, i, searches_per_thread)
                futures.append(future)
            
            # Wait for all threads
            for future in futures:
                future.result()
        
        total_time = time.time() - start_total
        
        # Calculate results
        if all_times:
            results = {
                "total_searches": num_threads * searches_per_thread,
                "successful_searches": successful_searches,
                "errors": errors,
                "success_rate": successful_searches / (num_threads * searches_per_thread),
                "total_time_seconds": total_time,
                "searches_per_second": successful_searches / total_time,
                "mean_time_ms": statistics.mean(all_times),
                "median_time_ms": statistics.median(all_times),
                "p95_time_ms": np.percentile(all_times, 95),
                "p99_time_ms": np.percentile(all_times, 99),
                "std_dev_ms": statistics.stdev(all_times) if len(all_times) > 1 else 0
            }
            
            print(f"\n📊 CONCURRENT SEARCH RESULTS:")
            print(f"Total searches: {results['total_searches']}")
            print(f"Successful: {results['successful_searches']}")
            print(f"Errors: {results['errors']}")
            print(f"Success rate: {results['success_rate']*100:.1f}%")
            print(f"Searches per second: {results['searches_per_second']:.1f}")
            print(f"Mean time: {results['mean_time_ms']:.3f} ms")
            print(f"P95 time: {results['p95_time_ms']:.3f} ms")
            print(f"P99 time: {results['p99_time_ms']:.3f} ms")
            
            return results
        else:
            print("❌ No successful concurrent searches!")
            return None
    
    def full_benchmark(self, num_insertions=100, num_searches=100, dimension=128):
        """Run complete benchmark suite"""
        print("🚀 RUNNING FULL DATABASE BENCHMARK")
        print("="*50)
        
        # Get initial database state
        db_info = self.face_db.get_database_info()
        print(f"Initial database size: {db_info.get('count', 0)} faces")
        
        # Run benchmarks
        insertion_results = self.benchmark_insertion(num_insertions, dimension)
        search_results = self.benchmark_search(num_searches, dimension=dimension)
        concurrent_results = self.benchmark_concurrent_search(dimension=dimension)
        
        # Final summary
        print(f"\n🏁 FINAL SUMMARY")
        print("="*50)
        
        if insertion_results:
            print(f"Insertion: {insertion_results['mean_time_ms']:.3f}ms avg")
        
        if search_results:
            print(f"Search: {search_results['mean_time_ms']:.3f}ms avg")
            
            # Overall assessment
            total_time = (insertion_results['mean_time_ms'] if insertion_results else 0) + search_results['mean_time_ms']
            print(f"Combined (insert + search): {total_time:.3f}ms")
            
            if total_time < 15:
                print(f"✅ OVERALL: PASSED (<15ms requirement)")
            else:
                print(f"❌ OVERALL: FAILED (>{total_time:.1f}ms)")
        
        if concurrent_results:
            print(f"Concurrent throughput: {concurrent_results['searches_per_second']:.1f} searches/sec")
        
        return {
            "insertion": insertion_results,
            "search": search_results,
            "concurrent": concurrent_results
        }


def main():
    parser = argparse.ArgumentParser(description='Database Performance Benchmark')
    parser.add_argument('--db_path', type=str, default='./face_db/chroma_data', help='Database path')
    parser.add_argument('--insertions', type=int, default=100, help='Number of insertions to test')
    parser.add_argument('--searches', type=int, default=100, help='Number of searches to test')
    parser.add_argument('--dimension', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--concurrent', action='store_true', help='Run concurrent search test')
    parser.add_argument('--insertion_only', action='store_true', help='Run insertion benchmark only')
    parser.add_argument('--search_only', action='store_true', help='Run search benchmark only')
    
    args = parser.parse_args()
    
    # Initialize benchmark
    benchmark = DatabaseBenchmark(args.db_path)
    
    # Run specified benchmarks
    if args.insertion_only:
        benchmark.benchmark_insertion(args.insertions, args.dimension)
    elif args.search_only:
        benchmark.benchmark_search(args.searches, dimension=args.dimension)
    elif args.concurrent:
        benchmark.benchmark_concurrent_search(dimension=args.dimension)
    else:
        # Run full benchmark
        benchmark.full_benchmark(args.insertions, args.searches, args.dimension)


if __name__ == "__main__":
    main()