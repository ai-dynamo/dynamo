#!/usr/bin/env python3
"""
Fixed OpenTelemetry Metrics Client
A Python client using the OpenTelemetry SDK to send metrics to OTLP endpoints
with Bearer token authentication, based on TypeScript reference implementation.
"""

import time
import os
import random
import re
from typing import Optional, Dict, Any, Union, Callable
from urllib.parse import urlparse

from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes


def mask_sensitive_urls(error_msg: str, url: str) -> str:
    """Comprehensively mask sensitive URLs and hostnames in error messages"""
    if not url:
        return error_msg
        
    try:
        parsed_url = urlparse(url)
        hostname = parsed_url.hostname
        path = parsed_url.path
        
        # Replace components in order of specificity
        if hostname:
            error_msg = error_msg.replace(hostname, "***HOSTNAME***")
        if url in error_msg:
            error_msg = error_msg.replace(url, "***OTLP_ENDPOINT***")
        if path and path in error_msg:
            error_msg = error_msg.replace(path, "***PATH***")
            
        # Also mask any remaining URL patterns
        if hostname:
            pattern = rf"https?://{re.escape(hostname)}"
            error_msg = re.sub(pattern, "***MASKED_URL***", error_msg)
            
    except Exception:
        # If URL parsing fails, do basic masking
        if url in error_msg:
            error_msg = error_msg.replace(url, "***OTLP_ENDPOINT***")
    
    return error_msg


class OTelMetricsClient:
    """Fixed OpenTelemetry Metrics Client with Bearer token authentication."""
    
    # Default endpoints (full URLs including /v1/metrics)
    PROD_ENDPOINT = os.getenv('OTLP_ENDPOINT')
    
    def __init__(
        self,
        endpoint: str = PROD_ENDPOINT,
        auth_token: str = None,
        service_name: str = "test-svc",
        service_version: str = "1.0.0",
        meter_name: str = "demo-metrics",
        export_interval_millis: int = 60000,
        auth_resource_key: str = "authorization"
    ):
        """
        Initialize the OpenTelemetry Metrics Client.
        
        Args:
            endpoint: OTLP HTTP endpoint URL (full URL including /v1/metrics)
            auth_token: Bearer token for authentication
            service_name: Service name for resource attributes
            service_version: Service version
            meter_name: Meter name/scope
            export_interval_millis: Export interval in milliseconds (default: 60s like TypeScript)
            auth_resource_key: Resource key for authentication (default: "authorization")
        """
        self.endpoint = endpoint
        self.auth_token = auth_token or os.getenv("OTEL_AUTH_TOKEN")
        self.service_name = service_name
        self.service_version = service_version
        self.meter_name = meter_name
        self.export_interval_millis = export_interval_millis
        self.auth_resource_key = auth_resource_key
        
        if not self.auth_token:
            raise ValueError("auth_token must be provided or set OTEL_AUTH_TOKEN environment variable")
        
        # Setup provider and meter
        self._setup_meter_provider()
        self._meter = metrics.get_meter(self.meter_name, self.service_version)
        
        # Cache for created instruments to avoid recreation
        self._instruments = {}
    
    def _setup_meter_provider(self):
        """Setup the OpenTelemetry meter provider with OTLP exporter."""
        
        # Create resource with service information and auth token
        # Following TypeScript pattern of including auth in resource
        resource_attributes = {
            ResourceAttributes.SERVICE_NAME: self.service_name,
            ResourceAttributes.SERVICE_VERSION: self.service_version,
            "deployment.environment": os.getenv("NODE_ENV", "production"),
            "host.name": os.getenv("HOSTNAME", "python-host"),
            "process.pid": str(os.getpid()),
        }
        
        # Add auth token to resource (like TypeScript implementation)
        resource_attributes[self.auth_resource_key] = self.auth_token
        
        resource = Resource.create(resource_attributes)
        
        # Setup headers with Bearer token (dual approach for compatibility)
        headers = {
            "Authorization": f"Bearer {self.auth_token}",
        }
        
        # Create OTLP Metric Exporter
        otlp_exporter = OTLPMetricExporter(
            endpoint=self.endpoint,
            headers=headers,
            timeout=30
        )
        
        # Create metric reader with periodic export (matching TypeScript interval)
        metric_reader = PeriodicExportingMetricReader(
            exporter=otlp_exporter,
            export_interval_millis=self.export_interval_millis
        )
        
        # Create Meter Provider
        self.meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[metric_reader]
        )
        
        # Set the global meter provider
        metrics.set_meter_provider(self.meter_provider)
    
    def create_counter(
        self,
        name: str,
        description: str = "",
        unit: str = "1"
    ):
        """
        Create or get a counter instrument (like TypeScript createCounter).
        
        Args:
            name: Counter name
            description: Counter description
            unit: Counter unit
            
        Returns:
            Counter instrument
        """
        instrument_key = f"counter_{name}"
        if instrument_key not in self._instruments:
            self._instruments[instrument_key] = self._meter.create_counter(
                name=name,
                description=description,
                unit=unit
            )
        return self._instruments[instrument_key]
    
    def create_up_down_counter(
        self,
        name: str,
        description: str = "",
        unit: str = "1"
    ):
        """
        Create or get an up-down counter instrument (like TypeScript createUpDownCounter).
        
        Args:
            name: Counter name
            description: Counter description
            unit: Counter unit
            
        Returns:
            UpDownCounter instrument
        """
        instrument_key = f"updowncounter_{name}"
        if instrument_key not in self._instruments:
            self._instruments[instrument_key] = self._meter.create_up_down_counter(
                name=name,
                description=description,
                unit=unit
            )
        return self._instruments[instrument_key]
    
    def create_histogram(
        self,
        name: str,
        description: str = "",
        unit: str = "1"
    ):
        """
        Create or get a histogram instrument (like TypeScript createHistogram).
        
        Args:
            name: Histogram name
            description: Histogram description
            unit: Histogram unit
            
        Returns:
            Histogram instrument
        """
        instrument_key = f"histogram_{name}"
        if instrument_key not in self._instruments:
            self._instruments[instrument_key] = self._meter.create_histogram(
                name=name,
                description=description,
                unit=unit
            )
        return self._instruments[instrument_key]
    
    def create_observable_gauge(
        self,
        name: str,
        description: str = "",
        unit: str = "1",
        callbacks: Optional[list] = None
    ):
        """
        Create or get an observable gauge instrument (like TypeScript createObservableGauge).
        
        Args:
            name: Gauge name
            description: Gauge description
            unit: Gauge unit
            callbacks: Optional list of callback functions
            
        Returns:
            ObservableGauge instrument
        """
        instrument_key = f"gauge_{name}"
        if instrument_key not in self._instruments:
            gauge = self._meter.create_observable_gauge(
                name=name,
                description=description,
                unit=unit,
                callbacks=callbacks or []
            )
            self._instruments[instrument_key] = gauge
        return self._instruments[instrument_key]
    
    def add_counter(
        self,
        name: str,
        value: Union[int, float],
        attributes: Optional[Dict[str, str]] = None,
        description: str = "",
        unit: str = "1"
    ):
        """
        Add value to a counter (like TypeScript counter.add()).
        
        Args:
            name: Counter name
            value: Value to add
            attributes: Additional attributes/labels
            description: Counter description
            unit: Counter unit
        """
        counter = self.create_counter(name, description, unit)
        counter.add(value, attributes or {})
    
    def add_up_down_counter(
        self,
        name: str,
        value: Union[int, float],
        attributes: Optional[Dict[str, str]] = None,
        description: str = "",
        unit: str = "1"
    ):
        """
        Add value to an up-down counter (like TypeScript upDownCounter.add()).
        
        Args:
            name: Counter name
            value: Value to add (can be negative)
            attributes: Additional attributes/labels
            description: Counter description
            unit: Counter unit
        """
        counter = self.create_up_down_counter(name, description, unit)
        counter.add(value, attributes or {})
    
    def record_histogram(
        self,
        name: str,
        value: Union[int, float],
        attributes: Optional[Dict[str, str]] = None,
        description: str = "",
        unit: str = "1"
    ):
        """
        Record a histogram value (like TypeScript histogram.record()).
        
        Args:
            name: Histogram name
            value: Value to record
            attributes: Additional attributes/labels
            description: Histogram description
            unit: Histogram unit
        """
        histogram = self.create_histogram(name, description, unit)
        histogram.record(value, attributes or {})
    
    def observe_gauge(
        self,
        name: str,
        callback: Callable,
        description: str = "",
        unit: str = "1"
    ):
        """
        Create a gauge with callback (like TypeScript gauge.addCallback()).
        
        Args:
            name: Gauge name
            callback: Callback function that provides the value
            description: Gauge description
            unit: Gauge unit
        """
        self.create_observable_gauge(name, description, unit, [callback])
    
    def flush(self):
        """Force flush all pending metrics."""
        try:
            meter_provider = metrics.get_meter_provider()
            if hasattr(meter_provider, 'force_flush'):
                meter_provider.force_flush(timeout_millis=30000)
                return True
        except Exception as e:
            masked_error = mask_sensitive_urls(str(e), self.endpoint)
            print(f"Warning: Could not flush metrics: {masked_error}")
            return False
    
    def shutdown(self):
        """Shutdown the meter provider and cleanup resources."""
        try:
            if hasattr(self.meter_provider, 'shutdown'):
                self.meter_provider.shutdown()
                return True
        except Exception as e:
            masked_error = mask_sensitive_urls(str(e), self.endpoint)
            print(f"Warning: Could not shutdown meter provider: {masked_error}")
            return False
