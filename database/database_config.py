"""
Database Configuration
Centralized configuration for database operations and performance tuning
"""

import os
from typing import Dict, Any

class DatabaseConfig:
    """Database configuration management"""
    
    # Default database settings
    DEFAULT_DB_PATH = "./face_db/chroma_data"
    DEFAULT_COLLECTION_NAME = "face_embeddings"
    
    # Performance settings for <15ms requirement
    HNSW_CONFIG = {
        # HNSW (Hierarchical Navigable Small World) parameters
        "hnsw:space": "cosine",        # Distance metric (cosine similarity)
        "hnsw:M": 16,                  # Number of connections per node (default: 16)
        "hnsw:ef_construction": 200,   # Search width during index construction (higher = better accuracy, slower build)
        "hnsw:ef": 100,                # Search width during query (higher = better accuracy, slower search)
        "hnsw:max_elements": 10000,    # Maximum number of elements (can be increased)
    }
    
    # Search parameters
    SEARCH_CONFIG = {
        "default_top_k": 5,            # Default number of results to return
        "default_threshold": 0.7,      # Default similarity threshold
        "max_top_k": 50,              # Maximum allowed top_k
        "min_threshold": 0.1,         # Minimum similarity threshold
        "max_threshold": 1.0,         # Maximum similarity threshold
    }
    
    # Embedding configuration
    EMBEDDING_CONFIG = {
        "dimension": 128,              # Embedding dimension (MobileFaceNet)
        "normalize": True,             # Whether to L2 normalize embeddings
        "dtype": "float32",            # Data type for embeddings
    }
    
    # Performance targets
    PERFORMANCE_TARGETS = {
        "max_search_latency_ms": 15.0,    # Maximum search latency
        "max_insert_latency_ms": 10.0,    # Maximum insert latency
        "target_throughput_qps": 100,     # Target queries per second
        "max_memory_usage_mb": 1024,      # Maximum memory usage
    }
    
    # Environment-specific configurations
    ENVIRONMENTS = {
        "development": {
            "db_path": "./face_db/dev_data",
            "log_level": "DEBUG",
            "enable_metrics": True,
            "backup_enabled": False,
        },
        "testing": {
            "db_path": "./face_db/test_data",
            "log_level": "INFO",
            "enable_metrics": True,
            "backup_enabled": False,
        },
        "production": {
            "db_path": "./face_db/prod_data",
            "log_level": "ERROR",
            "enable_metrics": True,
            "backup_enabled": True,
            "backup_interval_hours": 24,
        }
    }
    
    @classmethod
    def get_config(cls, environment: str = "development") -> Dict[str, Any]:
        """
        Get configuration for specified environment
        
        Args:
            environment (str): Environment name ('development', 'testing', 'production')
            
        Returns:
            dict: Complete configuration dictionary
        """
        # Start with default config
        config = {
            "database": {
                "db_path": cls.DEFAULT_DB_PATH,
                "collection_name": cls.DEFAULT_COLLECTION_NAME,
                "hnsw_config": cls.HNSW_CONFIG.copy(),
            },
            "search": cls.SEARCH_CONFIG.copy(),
            "embedding": cls.EMBEDDING_CONFIG.copy(),
            "performance": cls.PERFORMANCE_TARGETS.copy(),
        }
        
        # Apply environment-specific overrides
        if environment in cls.ENVIRONMENTS:
            env_config = cls.ENVIRONMENTS[environment]
            config.update(env_config)
            
            # Update database path if specified in environment
            if "db_path" in env_config:
                config["database"]["db_path"] = env_config["db_path"]
        
        # Override with environment variables if present
        config = cls._apply_environment_variables(config)
        
        return config
    
    @classmethod
    def _apply_environment_variables(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides"""
        
        # Database settings
        if "FACE_DB_PATH" in os.environ:
            config["database"]["db_path"] = os.environ["FACE_DB_PATH"]
        
        if "FACE_DB_COLLECTION" in os.environ:
            config["database"]["collection_name"] = os.environ["FACE_DB_COLLECTION"]
        
        # Performance settings
        if "FACE_DB_MAX_SEARCH_LATENCY" in os.environ:
            config["performance"]["max_search_latency_ms"] = float(os.environ["FACE_DB_MAX_SEARCH_LATENCY"])
        
        if "FACE_DB_MAX_INSERT_LATENCY" in os.environ:
            config["performance"]["max_insert_latency_ms"] = float(os.environ["FACE_DB_MAX_INSERT_LATENCY"])
        
        # HNSW tuning
        if "FACE_DB_HNSW_M" in os.environ:
            config["database"]["hnsw_config"]["hnsw:M"] = int(os.environ["FACE_DB_HNSW_M"])
        
        if "FACE_DB_HNSW_EF" in os.environ:
            config["database"]["hnsw_config"]["hnsw:ef"] = int(os.environ["FACE_DB_HNSW_EF"])
        
        if "FACE_DB_HNSW_EF_CONSTRUCTION" in os.environ:
            config["database"]["hnsw_config"]["hnsw:ef_construction"] = int(os.environ["FACE_DB_HNSW_EF_CONSTRUCTION"])
        
        return config
    
    @classmethod
    def get_optimized_config_for_latency(cls) -> Dict[str, Any]:
        """
        Get configuration optimized for <15ms latency requirement
        
        Returns:
            dict: Optimized configuration
        """
        config = cls.get_config("production")
        
        # Optimize HNSW parameters for speed
        config["database"]["hnsw_config"].update({
            "hnsw:M": 8,                    # Fewer connections = faster search
            "hnsw:ef": 50,                  # Lower ef = faster search  
            "hnsw:ef_construction": 100,    # Lower construction = faster build
        })
        
        # Adjust search parameters
        config["search"].update({
            "default_top_k": 3,             # Fewer results = faster
            "default_threshold": 0.75,      # Higher threshold = fewer results
        })
        
        return config
    
    @classmethod
    def get_optimized_config_for_accuracy(cls) -> Dict[str, Any]:
        """
        Get configuration optimized for accuracy (at cost of speed)
        
        Returns:
            dict: Accuracy-optimized configuration
        """
        config = cls.get_config("production")
        
        # Optimize HNSW parameters for accuracy
        config["database"]["hnsw_config"].update({
            "hnsw:M": 32,                   # More connections = better accuracy
            "hnsw:ef": 200,                 # Higher ef = better accuracy
            "hnsw:ef_construction": 400,    # Higher construction = better accuracy
        })
        
        # Adjust search parameters
        config["search"].update({
            "default_top_k": 10,            # More results
            "default_threshold": 0.6,       # Lower threshold = more results
        })
        
        return config
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> bool:
        """
        Validate configuration settings
        
        Args:
            config (dict): Configuration to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Check required keys
            required_keys = ["database", "search", "embedding", "performance"]
            for key in required_keys:
                if key not in config:
                    print(f"Missing required config key: {key}")
                    return False
            
            # Validate embedding dimension
            if config["embedding"]["dimension"] not in [128, 256, 512]:
                print(f"Invalid embedding dimension: {config['embedding']['dimension']}")
                return False
            
            # Validate search thresholds
            search_config = config["search"]
            if not (0.0 <= search_config["default_threshold"] <= 1.0):
                print(f"Invalid default threshold: {search_config['default_threshold']}")
                return False
            
            # Validate HNSW parameters
            hnsw_config = config["database"]["hnsw_config"]
            if hnsw_config["hnsw:M"] < 2 or hnsw_config["hnsw:M"] > 100:
                print(f"Invalid hnsw:M value: {hnsw_config['hnsw:M']}")
                return False
            
            if hnsw_config["hnsw:ef"] < 1:
                print(f"Invalid hnsw:ef value: {hnsw_config['hnsw:ef']}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Config validation error: {e}")
            return False
    
    @classmethod
    def print_config_summary(cls, config: Dict[str, Any]):
        """Print a summary of the configuration"""
        print(" DATABASE CONFIGURATION SUMMARY")
        print("=" * 40)
        
        # Database settings
        db_config = config["database"]
        print(f"Database Path: {db_config['db_path']}")
        print(f"Collection: {db_config['collection_name']}")
        
        # HNSW settings
        hnsw = db_config["hnsw_config"]
        print(f"HNSW M: {hnsw['hnsw:M']}")
        print(f"HNSW ef: {hnsw['hnsw:ef']}")
        print(f"HNSW ef_construction: {hnsw['hnsw:ef_construction']}")
        
        # Performance targets
        perf = config["performance"]
        print(f"Max Search Latency: {perf['max_search_latency_ms']}ms")
        print(f"Max Insert Latency: {perf['max_insert_latency_ms']}ms")
        print(f"Target Throughput: {perf['target_throughput_qps']} QPS")
        
        # Embedding settings
        embed = config["embedding"]
        print(f"Embedding Dimension: {embed['dimension']}")
        print(f"Normalization: {embed['normalize']}")
        
        print("=" * 40)


# Pre-defined configurations for easy access
DEVELOPMENT_CONFIG = DatabaseConfig.get_config("development")
TESTING_CONFIG = DatabaseConfig.get_config("testing")
PRODUCTION_CONFIG = DatabaseConfig.get_config("production")
LATENCY_OPTIMIZED_CONFIG = DatabaseConfig.get_optimized_config_for_latency()
ACCURACY_OPTIMIZED_CONFIG = DatabaseConfig.get_optimized_config_for_accuracy()


def main():
    """Demo script to show configuration usage"""
    print(" DATABASE CONFIGURATION DEMO")
    print()
    
    # Show different configurations
    configs = {
        "Development": DEVELOPMENT_CONFIG,
        "Production": PRODUCTION_CONFIG,
        "Latency Optimized": LATENCY_OPTIMIZED_CONFIG,
        "Accuracy Optimized": ACCURACY_OPTIMIZED_CONFIG,
    }
    
    for name, config in configs.items():
        print(f"\n{name} Configuration:")
        DatabaseConfig.print_config_summary(config)
        print(f"Valid: {DatabaseConfig.validate_config(config)}")


if __name__ == "__main__":
    main()