#!/usr/bin/env python3
"""
Worker Profile Client for Dynamo CI
A client class to interact with the Redshift database for worker profile data.
"""

import time
import traceback
import boto3
import logging
import json
import os
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class WorkerProfileDataPoint:
    """Data class for worker profile data points"""
    model_name: str
    model_hf_name: Optional[str] = None
    cluster_name: Optional[str] = None
    gpu_type: Optional[str] = None
    gpu_count: Optional[int] = None
    node_count: Optional[int] = None
    backend: Optional[str] = None
    worker_type: Optional[str] = None
    tp: Optional[int] = None
    dp: Optional[int] = None
    pp: Optional[int] = None
    dynamo_init_command: Optional[str] = None
    worker_init_command: Optional[str] = None
    decode_osl: Optional[float] = None
    max_context_length: Optional[int] = None
    max_kv_tokens: Optional[int] = None
    profiler_command: Optional[str] = None
    prefill_isl: Optional[float] = None
    prefill_ttft: Optional[float] = None
    prefill_throughput_per_gpu: Optional[float] = None
    x_kv_usage: Optional[float] = None
    y_context_length: Optional[int] = None
    z_itl: Optional[float] = None
    z_thpt_per_gpu: Optional[float] = None
    graphs: Optional[str] = None
    uniq_profile_id: Optional[str] = None
    ci_pipeline_id: Optional[str] = None
    ci_job_id: Optional[str] = None
    image_name: Optional[str] = None
    image_hash: Optional[str] = None
    dynamo_version: Optional[str] = None
    dynamo_commit: Optional[str] = None
    node_id: Optional[str] = None
    updated_at: Optional[datetime] = None


class WorkerProfileClient:
    """
    Client class for interacting with the Redshift database for worker profile data.
    """
    
    def __init__(self, 
                 workgroup_name: str = "default-workgroup",
                 database_name: str = "dynamo_worker_profile",
                 table_name: str = "public.dynamo_worker_data_points",
                 region_name: str = "us-west-2"):
        """
        Initialize the WorkerProfileClient.
        
        Args:
            workgroup_name: Name of the Redshift workgroup
            database_name: Name of the Redshift database
            table_name: Name of the Redshift table
            region_name: AWS region name
        """
        self.workgroup_name = workgroup_name
        self.database_name = database_name
        self.table_name = table_name
        self.region_name = region_name
        
        # Initialize Redshift Data API client
        self.redshift_data_api_client = boto3.client('redshift-data', region_name=region_name)
        
        # Verify AWS credentials
        try:
            sts = boto3.client('sts')
            sts.get_caller_identity()
            logger.info("AWS credentials verified successfully")
        except Exception as e:
            logger.error(f"Failed to verify AWS credentials: {e}")
            raise
    
    def _execute_sql_query(self, query: str, is_synchronous: bool = True) -> Dict:
        """
        Execute SQL query using Redshift Data API.
        
        Args:
            query: SQL query to execute
            is_synchronous: Whether to wait for query completion
            
        Returns:
            Dict containing query results and metadata
        """
        MAX_WAIT_CYCLES = 60  # Increased for longer queries
        attempts = 0
        
        try:
            # Execute the statement
            response = self.redshift_data_api_client.execute_statement(
                Database=self.database_name,
                WorkgroupName=self.workgroup_name,
                Sql=query
            )
            query_id = response["Id"]
            
            logger.info(f"Executing query with ID: {query_id}")
            logger.info(f"Query: {query}")
            
            if not is_synchronous:
                return {"query_id": query_id, "status": "STARTED"}
            
            # Wait for completion
            while attempts < MAX_WAIT_CYCLES:
                attempts += 1
                time.sleep(2)  # Increased sleep time
                
                desc = self.redshift_data_api_client.describe_statement(Id=query_id)
                query_status = desc["Status"]
                
                if query_status == "FAILED":
                    error_msg = desc.get("Error", "Unknown error")
                    raise Exception(f'SQL query failed: {query_id}: {error_msg}')
                
                elif query_status == "FINISHED":
                    logger.info(f"Query completed successfully: {query_id}")
                    result = {"query_id": query_id, "status": query_status}
                    
                    # Get results if available
                    if desc.get('HasResultSet', False):
                        result_response = self.redshift_data_api_client.get_statement_result(Id=query_id)
                        result["records"] = result_response.get('Records', [])
                        result["column_metadata"] = result_response.get('ColumnMetadata', [])
                    
                    return result
                
                else:
                    logger.debug(f"Query status: {query_status} (attempt {attempts})")
            
            # Timeout
            raise Exception(f"Query timed out after {MAX_WAIT_CYCLES} attempts")
            
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            logger.error(traceback.format_exc())
            raise

    def create_table(self):
        """
        Create the worker profile table.
        """
        query = f"""
       CREATE TABLE {self.table_name} (
    model_name character varying(100) NOT NULL ENCODE lzo,
    model_hf_name character varying(200) ENCODE lzo,
    cluster_name character varying(100) ENCODE lzo,
    gpu_type character varying(50) ENCODE lzo,
    gpu_count integer ENCODE az64,
    node_count integer ENCODE az64,
    backend character varying(50) ENCODE lzo,
    worker_type character varying(50) ENCODE lzo,
    tp integer ENCODE az64,
    dp integer ENCODE az64,
    pp integer ENCODE az64,
    dynamo_init_command character varying(500) ENCODE lzo,
    worker_init_command character varying(500) ENCODE lzo,
    decode_osl numeric(10, 2) ENCODE az64,
    max_context_length integer ENCODE az64,
    max_kv_tokens integer ENCODE az64,
    profiler_command character varying(500) ENCODE lzo,
    prefill_isl numeric(10, 2) ENCODE az64,
    prefill_ttft numeric(10, 2) ENCODE az64,
    prefill_throughput_per_gpu numeric(10, 2) ENCODE az64,
    x_kv_usage numeric(10, 2) ENCODE az64,
    y_context_length integer ENCODE az64,
    z_itl numeric(10, 2) ENCODE az64,
    z_thpt_per_gpu numeric(10, 2) ENCODE az64,
    graphs character varying(1000) ENCODE lzo,
    uniq_profile_id character varying(50) ENCODE lzo,
    ci_pipeline_id character varying(50) ENCODE lzo,
    ci_job_id character varying(50) ENCODE lzo,
    image_name character varying(200) ENCODE lzo,
    image_hash character varying(64) ENCODE lzo,
    dynamo_version character varying(50) ENCODE lzo,
    dynamo_commit character varying(64) ENCODE lzo,
    node_id character varying(50) ENCODE lzo,
    updated_at timestamp without time zone DEFAULT getdate() ENCODE az64
) DISTSTYLE AUTO;
        """
        self._execute_sql_query(query)

    def submit_data_point(self, data_point: WorkerProfileDataPoint) -> bool:
        """
        Submit a data point to the worker profile table.
        
        Args:
            data_point: WorkerProfileDataPoint object containing the data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Convert data point to dictionary, excluding None values
            data_dict = {k: v for k, v in data_point.__dict__.items() if v is not None}
            
            # Handle datetime objects
            if 'updated_at' in data_dict and isinstance(data_dict['updated_at'], datetime):
                data_dict['updated_at'] = data_dict['updated_at'].isoformat()
            
            # Build the INSERT query
            columns = list(data_dict.keys())
            values = []
            
            for value in data_dict.values():
                if isinstance(value, str):
                    # Escape single quotes in strings
                    escaped_value = value.replace("'", "''")
                    values.append(f"'{escaped_value}'")
                elif isinstance(value, (int, float)):
                    values.append(str(value))
                else:
                    values.append(f"'{str(value)}'")
            
            query = f"""
            INSERT INTO {self.table_name} ({', '.join(columns)})
            VALUES ({', '.join(values)})
            """
            
            result = self._execute_sql_query(query)
            logger.info(f"Successfully submitted data point for model: {data_point.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit data point: {e}")
            return False
    
    def _build_insert_query(self, **kwargs) -> str:
        """
        Build an INSERT query from keyword arguments.
        
        Args:
            **kwargs: Field names and values for the INSERT statement
            
        Returns:
            str: SQL INSERT statement
        """
        # Filter out None values
        data_dict = {k: v for k, v in kwargs.items() if v is not None}
        
        # Build the INSERT query
        columns = list(data_dict.keys())
        values = []
        
        for value in data_dict.values():
            if isinstance(value, str):
                # Check if it's a SQL function (like CURRENT_TIMESTAMP)
                if value.upper() in ['CURRENT_TIMESTAMP', 'NOW()', 'GETDATE()']:
                    values.append(value)
                else:
                    # Escape single quotes in strings
                    escaped_value = value.replace("'", "''")
                    values.append(f"'{escaped_value}'")
            elif isinstance(value, (int, float)):
                values.append(str(value))
            else:
                values.append(f"'{str(value)}'")
        
        query = f"""
        INSERT INTO {self.table_name} ({', '.join(columns)})
        VALUES ({', '.join(values)})
        """
        
        return query
    
    def _execute_batch_statements(self, statements: List[str]) -> Dict[str, int]:
        """
        Execute multiple SQL statements in batch using batch_execute_statement.
        
        Args:
            statements: List of SQL statements to execute
            
        Returns:
            Dict with counts of successful and failed executions
        """
        results = {"successful": 0, "failed": 0}
        
        if not statements:
            return results
        
        try:
            logger.info(f"Executing batch of {len(statements)} statements")
            
            # Execute batch statements
            response = self.redshift_data_api_client.batch_execute_statement(
                Database=self.database_name,
                WorkgroupName=self.workgroup_name,
                Sqls=statements
            )
            batch_id = response["Id"]
            
            logger.info(f"Batch execution started with ID: {batch_id}")
            
            # Wait for completion
            MAX_WAIT_CYCLES = 120  # Increased for batch operations
            attempts = 0
            
            while attempts < MAX_WAIT_CYCLES:
                attempts += 1
                time.sleep(3)  # Increased sleep time for batch operations
                
                desc = self.redshift_data_api_client.describe_statement(Id=batch_id)
                batch_status = desc["Status"]
                
                if batch_status == "FAILED":
                    error_msg = desc.get("Error", "Unknown error")
                    logger.error(f'Batch execution failed: {batch_id}: {error_msg}')
                    results["failed"] = len(statements)
                    break
                
                elif batch_status == "FINISHED":
                    logger.info(f"Batch execution completed successfully: {batch_id}")
                    
                    # For batch operations, we need to check the sub-statements from describe_statement
                    # rather than trying to get results with the batch ID
                    desc = self.redshift_data_api_client.describe_statement(Id=batch_id)
                    
                    # Count successful and failed statements from sub-statements
                    if "SubStatements" in desc:
                        for sub_statement in desc["SubStatements"]:
                            sub_status = sub_statement.get("Status")
                            if sub_status == "FINISHED":
                                results["successful"] += 1
                            else:
                                results["failed"] += 1
                                error_msg = sub_statement.get("Error", "Unknown error")
                                logger.error(f"Sub-statement failed: {error_msg}")
                    else:
                        # If no sub-statements info, assume all succeeded
                        results["successful"] = len(statements)
                    
                    break
                
                else:
                    logger.debug(f"Batch status: {batch_status} (attempt {attempts})")
            
            if attempts >= MAX_WAIT_CYCLES:
                logger.error(f"Batch execution timed out after {MAX_WAIT_CYCLES} attempts")
                results["failed"] = len(statements)
            
            logger.info(f"Batch execution results: {results['successful']} successful, {results['failed']} failed")
            return results
            
        except Exception as e:
            logger.error(f"Error executing batch statements: {e}")
            logger.error(traceback.format_exc())
            results["failed"] = len(statements)
            return results
    
    def get_engine_configurations(self, 
            hf_model_name: str, 
            hardware_sku: str, 
            context_length: int, 
            mode: str = "p") -> List[str]:
        """
        Get all available engine configuration IDs that match the given input.
        
        Args:
            hf_model_name: HuggingFace model name
            hardware_sku: Hardware SKU (e.g., "h100")
            context_length: Context length
            mode: Mode ("p" for prefill or "d" for decode)
            
        Returns:
            List of configuration IDs
        """
        try:
            # Map mode to worker_type
            worker_type_map = {"p": "prefill", "d": "decode"}
            worker_type = worker_type_map.get(mode.lower(), "prefill")
            
            query = f"""
            SELECT DISTINCT uniq_profile_id
            FROM {self.table_name}
            WHERE model_hf_name = '{hf_model_name.replace("'", "''")}'
            AND gpu_type = '{hardware_sku.replace("'", "''")}'
            AND max_context_length >= {context_length}
            AND worker_type = '{worker_type}'
            ORDER BY uniq_profile_id
            """
            
            result = self._execute_sql_query(query)
            
            if "records" in result:
                config_ids = []
                for record in result["records"]:
                    if record and len(record) > 0:
                        config_ids.append(str(record[0].get('stringValue', '')))
                return config_ids
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get engine configurations: {e}")
            return []
    
    def get_raw_data(self, 
                    model: str,
                    gpu_type: str,
                    worker_type: str,
                    backend: str,
                    backend_version: Optional[str] = None,
                    table_version: int = 0) -> Union[Tuple[List[float], List[float], List[float]], 
                                                   Tuple[List[float], List[int], List[float], List[float], List[int]]]:
        """
        Get raw data from the worker profile table.
        
        Args:
            model: HuggingFace model name
            gpu_type: GPU type (e.g., "h100")
            worker_type: Worker type ("prefill" or "decode")
            backend: Backend (e.g., "vllm", "sglang", "trtllm")
            backend_version: Backend version (defaults to max version)
            table_version: Table version (defaults to 0)
            
        Returns:
            Tuple of data arrays based on worker_type
        """
        try:
            # Build the base query
            base_query = f"""
            SELECT *
            FROM {self.table_name}
            WHERE model_hf_name = '{model.replace("'", "''")}'
            AND gpu_type = '{gpu_type.replace("'", "''")}'
            AND worker_type = '{worker_type.replace("'", "''")}'
            AND backend = '{backend.replace("'", "''")}'
            """
            
            if backend_version:
                base_query += f" AND backend_version = '{backend_version.replace("'", "''")}'"
            
            base_query += f" AND table_version = {table_version}"
            base_query += " ORDER BY updated_at DESC"
            
            result = self._execute_sql_query(base_query)
            
            if "records" not in result or not result["records"]:
                logger.warning(f"No data found for the specified criteria")
                return ([], [], []) if worker_type == "prefill" else ([], [], [], [], [])
            
            # Convert records to pandas DataFrame for easier processing
            columns = [col['name'] for col in result.get('column_metadata', [])]
            data = []
            
            for record in result["records"]:
                row = []
                for field in record:
                    if field.get('stringValue') is not None:
                        row.append(field['stringValue'])
                    elif field.get('longValue') is not None:
                        row.append(field['longValue'])
                    elif field.get('doubleValue') is not None:
                        row.append(field['doubleValue'])
                    else:
                        row.append(None)
                data.append(row)
            
            df = pd.DataFrame(data, columns=columns)
            
            if worker_type == "prefill":
                prefill_isl = df['prefill_isl'].dropna().astype(float).tolist()
                prefill_ttft = df['prefill_ttft'].dropna().astype(float).tolist()
                prefill_throughput_per_gpu = df['prefill_throughput_per_gpu'].dropna().astype(float).tolist()
                return prefill_isl, prefill_ttft, prefill_throughput_per_gpu
                
            elif worker_type == "decode":
                x_kv_usage = df['x_kv_usage'].dropna().astype(float).tolist()
                y_context_length = df['y_context_length'].dropna().astype(int).tolist()
                z_itl = df['z_itl'].dropna().astype(float).tolist()
                z_thpt_per_gpu = df['z_thpt_per_gpu'].dropna().astype(float).tolist()
                max_kv_tokens = df['max_kv_tokens'].dropna().astype(int).tolist()
                return x_kv_usage, y_context_length, z_itl, z_thpt_per_gpu, max_kv_tokens
                
            else:
                raise ValueError(f"Unsupported worker_type: {worker_type}")
                
        except Exception as e:
            logger.error(f"Failed to get raw data: {e}")
            return ([], [], []) if worker_type == "prefill" else ([], [], [], [], [])
    
    def get_profile_by_id(self, profile_id: str) -> Optional[WorkerProfileDataPoint]:
        """
        Get a specific profile by its unique ID.
        
        Args:
            profile_id: Unique profile ID
            
        Returns:
            WorkerProfileDataPoint or None if not found
        """
        try:
            query = f"""
            SELECT * FROM {self.table_name}
            WHERE uniq_profile_id = '{profile_id.replace("'", "''")}'
            """
            
            result = self._execute_sql_query(query)
            
            if "records" not in result or not result["records"]:
                return None
            
            # Convert the first record to WorkerProfileDataPoint
            record = result["records"][0]
            columns = [col['name'] for col in result.get('column_metadata', [])]
            
            data_dict = {}
            for i, field in enumerate(record):
                col_name = columns[i] if i < len(columns) else f"col_{i}"
                
                if field.get('stringValue') is not None:
                    data_dict[col_name] = field['stringValue']
                elif field.get('longValue') is not None:
                    data_dict[col_name] = field['longValue']
                elif field.get('doubleValue') is not None:
                    data_dict[col_name] = field['doubleValue']
                else:
                    data_dict[col_name] = None
            
            return WorkerProfileDataPoint(**data_dict)
            
        except Exception as e:
            logger.error(f"Failed to get profile by ID: {e}")
            return None
    
    def list_available_models(self) -> List[str]:
        """
        Get list of all available models in the database.
        
        Returns:
            List of model names
        """
        try:
            query = f"""
            SELECT DISTINCT model_hf_name
            FROM {self.table_name}
            WHERE model_hf_name IS NOT NULL
            ORDER BY model_hf_name
            """
            
            result = self._execute_sql_query(query)
            
            if "records" in result:
                models = []
                for record in result["records"]:
                    if record and len(record) > 0:
                        model_name = record[0].get('stringValue', '')
                        if model_name:
                            models.append(model_name)
                return models
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to list available models: {e}")
            return []
    
    def batch_submit_data_points(self, 
                                profile_config: WorkerProfileDataPoint,
                                prefill_raw_data: Optional[np.ndarray] = None,
                                decode_raw_data: Optional[np.ndarray] = None) -> Dict[str, int]:
        """
        Batch submit data points for both prefill and decode data.
        
        Args:
            profile_config: WorkerProfileDataPoint with common configuration fields
            prefill_raw_data: Numpy array containing prefill data with keys:
                             ['prefill_isl', 'prefill_ttft', 'prefill_thpt_per_gpu']
            decode_raw_data: Numpy array containing decode data with keys:
                            ['x_kv_usage', 'y_context_length', 'z_itl', 'z_thpt_per_gpu', 'max_kv_tokens']
        
        Returns:
            Dict with counts of successful and failed submissions
        """
        results = {"successful": 0, "failed": 0}
        
        try:
            # Process prefill data if provided
            if prefill_raw_data is not None:
                logger.info("Processing prefill data...")
                prefill_results = self._process_prefill_data(profile_config, prefill_raw_data)
                results["successful"] += prefill_results["successful"]
                results["failed"] += prefill_results["failed"]
            
            # Process decode data if provided
            if decode_raw_data is not None:
                logger.info("Processing decode data...")
                decode_results = self._process_decode_data(profile_config, decode_raw_data)
                results["successful"] += decode_results["successful"]
                results["failed"] += decode_results["failed"]
            
            logger.info(f"Batch submission completed. Successful: {results['successful']}, Failed: {results['failed']}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to batch submit data points: {e}")
            return results
    
    def _process_prefill_data(self, 
                             profile_config: WorkerProfileDataPoint,
                             prefill_raw_data: np.ndarray) -> Dict[str, int]:
        """
        Process and submit prefill data points using batch execution.
        
        Args:
            profile_config: Base configuration for data points
            prefill_raw_data: Numpy array with prefill data
            
        Returns:
            Dict with counts of successful and failed submissions
        """
        results = {"successful": 0, "failed": 0}
        
        try:
            # Extract arrays from the numpy data
            prefill_isl = prefill_raw_data['prefill_isl']
            prefill_ttft = prefill_raw_data['prefill_ttft']
            prefill_thpt_per_gpu = prefill_raw_data['prefill_thpt_per_gpu']
            
            # Ensure all arrays have the same length
            min_length = min(len(prefill_isl), len(prefill_ttft), len(prefill_thpt_per_gpu))
            
            logger.info(f"Processing {min_length} prefill data points using batch execution")
            
            # Prepare batch INSERT statements
            batch_statements = []
            
            for i in range(min_length):
                try:
                    # Create INSERT statement for each data point
                    insert_query = self._build_insert_query(
                        model_name=profile_config.model_name,
                        model_hf_name=profile_config.model_hf_name,
                        cluster_name=profile_config.cluster_name,
                        gpu_type=profile_config.gpu_type,
                        gpu_count=profile_config.gpu_count,
                        node_count=profile_config.node_count,
                        backend=profile_config.backend,
                        worker_type="prefill",  # Override for prefill data
                        tp=profile_config.tp,
                        dp=profile_config.dp,
                        pp=profile_config.pp,
                        dynamo_init_command=profile_config.dynamo_init_command,
                        worker_init_command=profile_config.worker_init_command,
                        decode_osl=profile_config.decode_osl,
                        max_context_length=profile_config.max_context_length,
                        max_kv_tokens=profile_config.max_kv_tokens,
                        profiler_command=profile_config.profiler_command,
                        # Prefill-specific data
                        prefill_isl=float(prefill_isl[i]),
                        prefill_ttft=float(prefill_ttft[i]),
                        prefill_throughput_per_gpu=float(prefill_thpt_per_gpu[i]),
                        # Other fields from profile_config
                        graphs=profile_config.graphs,
                        uniq_profile_id=profile_config.uniq_profile_id,
                        ci_pipeline_id=profile_config.ci_pipeline_id,
                        ci_job_id=profile_config.ci_job_id,
                        image_name=profile_config.image_name,
                        image_hash=profile_config.image_hash,
                        dynamo_version=profile_config.dynamo_version,
                        dynamo_commit=profile_config.dynamo_commit,
                        node_id=profile_config.node_id,
                        updated_at="CURRENT_TIMESTAMP"
                    )
                    
                    batch_statements.append(insert_query)
                        
                except Exception as e:
                    logger.error(f"Failed to prepare prefill data point {i}: {e}")
                    results["failed"] += 1
            
            # Execute batch statements
            if batch_statements:
                batch_results = self._execute_batch_statements(batch_statements)
                results["successful"] += batch_results["successful"]
                results["failed"] += batch_results["failed"]
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to process prefill data: {e}")
            return results
    
    def _process_decode_data(self, 
                            profile_config: WorkerProfileDataPoint,
                            decode_raw_data: np.ndarray) -> Dict[str, int]:
        """
        Process and submit decode data points using batch execution.
        
        Args:
            profile_config: Base configuration for data points
            decode_raw_data: Numpy array with decode data
            
        Returns:
            Dict with counts of successful and failed submissions
        """
        results = {"successful": 0, "failed": 0}
        
        try:
            # Extract arrays from the numpy data
            x_kv_usage = decode_raw_data['x_kv_usage']
            y_context_length = decode_raw_data['y_context_length']
            z_itl = decode_raw_data['z_itl']
            z_thpt_per_gpu = decode_raw_data['z_thpt_per_gpu']
            max_kv_tokens = decode_raw_data['max_kv_tokens']
            
            # max_kv_tokens might be a single value, handle it appropriately
            if len(max_kv_tokens) == 1:
                max_kv_tokens_value = int(max_kv_tokens[0])
            else:
                max_kv_tokens_value = int(max_kv_tokens[0])  # Use first value if multiple
            
            # Ensure all arrays have the same length (except max_kv_tokens)
            min_length = min(len(x_kv_usage), len(y_context_length), len(z_itl), len(z_thpt_per_gpu))
            
            logger.info(f"Processing {min_length} decode data points using batch execution")
            
            # Prepare batch INSERT statements
            batch_statements = []
            
            for i in range(min_length):
                try:
                    # Create INSERT statement for each data point
                    insert_query = self._build_insert_query(
                        model_name=profile_config.model_name,
                        model_hf_name=profile_config.model_hf_name,
                        cluster_name=profile_config.cluster_name,
                        gpu_type=profile_config.gpu_type,
                        gpu_count=profile_config.gpu_count,
                        node_count=profile_config.node_count,
                        backend=profile_config.backend,
                        worker_type="decode",  # Override for decode data
                        tp=profile_config.tp,
                        dp=profile_config.dp,
                        pp=profile_config.pp,
                        dynamo_init_command=profile_config.dynamo_init_command,
                        worker_init_command=profile_config.worker_init_command,
                        decode_osl=profile_config.decode_osl,
                        max_context_length=profile_config.max_context_length,
                        max_kv_tokens=max_kv_tokens_value,
                        profiler_command=profile_config.profiler_command,
                        # Decode-specific data
                        x_kv_usage=float(x_kv_usage[i]),
                        y_context_length=int(y_context_length[i]),
                        z_itl=float(z_itl[i]),
                        z_thpt_per_gpu=float(z_thpt_per_gpu[i]),
                        # Other fields from profile_config
                        graphs=profile_config.graphs,
                        uniq_profile_id=profile_config.uniq_profile_id,
                        ci_pipeline_id=profile_config.ci_pipeline_id,
                        ci_job_id=profile_config.ci_job_id,
                        image_name=profile_config.image_name,
                        image_hash=profile_config.image_hash,
                        dynamo_version=profile_config.dynamo_version,
                        dynamo_commit=profile_config.dynamo_commit,
                        node_id=profile_config.node_id,
                        updated_at="CURRENT_TIMESTAMP"
                    )
                    
                    batch_statements.append(insert_query)
                        
                except Exception as e:
                    logger.error(f"Failed to prepare decode data point {i}: {e}")
                    results["failed"] += 1
            
            # Execute batch statements
            if batch_statements:
                batch_results = self._execute_batch_statements(batch_statements)
                results["successful"] += batch_results["successful"]
                results["failed"] += batch_results["failed"]
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to process decode data: {e}")
            return results


def main():
    """
    Example usage of the WorkerProfileClient
    Only batch-submit is for real use.
    Other actions are for testing.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Worker Profile Client Example')
    parser.add_argument('--action', choices=['submit', 'query', 'list-models', 'batch-submit', 'create-table'], required=True)

    # Arguments for batch-submit action
    parser.add_argument('--model_name', type=str, help='Model name', required=False)
    parser.add_argument('--model_hf_name', type=str, help='HuggingFace model name', required=False)
    parser.add_argument('--gpu_type', type=str, help='GPU type (e.g., h100)', required=False)
    parser.add_argument('--backend', type=str, help='Backend (e.g., vllm)', required=False)
    parser.add_argument('--gpu_count', type=int, help='Number of GPUs', required=False)
    parser.add_argument('--node_count', type=int, help='Number of nodes', required=False)
    parser.add_argument('--mode', type=str, help='Mode (prefill or decode)', required=False)
    parser.add_argument('--max_context_length', type=int, help='Maximum context length', required=False)
    parser.add_argument('--max_kv_tokens', type=int, help='Maximum KV tokens', required=False)
    parser.add_argument('--tp', type=int, help='Tensor parallelism', required=False)
    parser.add_argument('--pp', type=int, help='Pipeline parallelism', required=False)
    parser.add_argument('--dp', type=int, help='Data parallelism', required=False)
    parser.add_argument('--raw_data_path', type=str, help='Path to raw data .npz file', required=False)
    parser.add_argument('--cluster_name', type=str, help='Cluster name', required=False)
    parser.add_argument('--uniq_profile_id', type=str, help='Unique profile ID', required=False)
    parser.add_argument('--ci_pipeline_id', type=str, help='CI pipeline ID', required=False)
    parser.add_argument('--ci_job_id', type=str, help='CI job ID', required=False)
    parser.add_argument('--node_id', type=str, help='Node ID', required=False)
    args = parser.parse_args()
    
    try:
        # Initialize client
        client = WorkerProfileClient()
        
        if args.action == 'list-models':
            models = client.list_available_models()
            print(f"Available models: {models}")
        elif args.action == 'create-table':
            client.create_table()
            print("Table created successfully")
        elif args.action == 'query':
            # Example query
            configs = client.get_engine_configurations(
                hf_model_name="meta-llama/Llama-2-7b-chat-hf",
                hardware_sku="h100",
                context_length=4096,
                mode="p"
            )
            print(f"Engine configurations: {configs}")

        elif args.action == 'batch-submit':
            # Validate required arguments
            required_args = ['model_name', 'model_hf_name', 'gpu_type', 'backend', 'gpu_count', 
                           'node_count', 'mode', 'max_context_length', 'max_kv_tokens', 
                           'tp', 'pp', 'dp', 'raw_data_path', 'cluster_name']
            
            missing_args = [arg for arg in required_args if getattr(args, arg) is None]
            if missing_args:
                logger.error(f"Missing required arguments for batch-submit: {missing_args}")
                return 1
            
            # Create profile configuration from command-line arguments
            profile_config = WorkerProfileDataPoint(
                model_name=args.model_name,
                model_hf_name=args.model_hf_name,
                gpu_type=args.gpu_type,
                backend=args.backend,
                gpu_count=args.gpu_count,
                node_count=args.node_count,
                max_context_length=args.max_context_length,
                max_kv_tokens=args.max_kv_tokens,
                tp=args.tp,
                pp=args.pp,
                dp=args.dp,
                # Add some default values for optional fields
                uniq_profile_id=args.uniq_profile_id or f"batch_{args.model_name}_{args.mode}_{int(time.time())}",
                ci_pipeline_id=args.ci_pipeline_id or os.environ.get('CI_PIPELINE_ID', 'unknown'),
                ci_job_id=args.ci_job_id or os.environ.get('CI_JOB_ID', 'unknown'),
                node_id=args.node_id or os.environ.get('NODE_ID', 'unknown'),
                cluster_name=args.cluster_name
            )
            
            # Load raw data from .npz file
            try:
                logger.info(f"Loading raw data from: {args.raw_data_path}")
                raw_data = np.load(args.raw_data_path)
                
                # Determine which data to process based on mode
                prefill_data = None
                decode_data = None
                
                if args.mode == "prefill":
                    # Check for prefill data keys
                    prefill_keys = ['prefill_isl', 'prefill_ttft', 'prefill_thpt_per_gpu']
                    if all(key in raw_data.keys() for key in prefill_keys):
                        prefill_data = {key: raw_data[key] for key in prefill_keys}
                        logger.info(f"Found prefill data with {len(prefill_data['prefill_isl'])} data points")
                    else:
                        logger.error(f"Missing prefill data keys in {args.raw_data_path}")
                        return 1
                        
                elif args.mode == "decode":
                    # Check for decode data keys
                    decode_keys = ['x_kv_usage', 'y_context_length', 'z_itl', 'z_thpt_per_gpu', 'max_kv_tokens']
                    if all(key in raw_data.keys() for key in decode_keys):
                        decode_data = {key: raw_data[key] for key in decode_keys}
                        logger.info(f"Found decode data with {len(decode_data['x_kv_usage'])} data points")
                    else:
                        logger.error(f"Missing decode data keys in {args.raw_data_path}")
                        return 1
                else:
                    logger.error(f"Unsupported mode: {args.mode}. Must be 'prefill' or 'decode'")
                    return 1
                
                # Submit batch data
                results = client.batch_submit_data_points(
                    profile_config=profile_config,
                    prefill_raw_data=prefill_data,
                    decode_raw_data=decode_data
                )
                
                print(f"Batch submission results: {results}")
                
                # Exit with error code if any failures
                if results["failed"] > 0:
                    logger.error(f"Batch submission had {results['failed']} failures")
                    return 1
                    
            except FileNotFoundError:
                logger.error(f"Raw data file not found: {args.raw_data_path}")
                return 1
            except Exception as e:
                logger.error(f"Error loading raw data: {e}")
                return 1
    
    except Exception as e:
        logger.error(f"Error in main: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
