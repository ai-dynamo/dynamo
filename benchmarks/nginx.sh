#!/bin/bash

# Create minimal nginx config for SSE passthrough
cat > /tmp/nginx-sse.conf << 'EOF'
worker_processes auto;
worker_rlimit_nofile 65535;

events {
    worker_connections 2048;
}

http {
    upstream sse_backends {
        server 127.0.0.1:8080;
        server 127.0.0.1:8090;
        # Keep connections alive to backends
        keepalive 100;
        keepalive_requests 10000;
        keepalive_time 1h;
    }

    server {
        listen 8070;
        
        location / {
            proxy_pass http://sse_backends;
            proxy_http_version 1.1;
            
            # Essential for SSE - maintain connection
            proxy_set_header Connection '';
            proxy_set_header Upgrade $http_upgrade;
            
            proxy_buffering off;
            proxy_cache off;
            
            # Preserve headers
            proxy_pass_request_headers on;
            
            # Disable response buffering at all levels
            proxy_max_temp_file_size 0;
            
            # SSE specific settings
            chunked_transfer_encoding off;
            proxy_set_header X-Accel-Buffering no;
            
            # Timeouts - all very high to prevent premature closure
            proxy_read_timeout 3600s;
            proxy_connect_timeout 3600s;
            proxy_send_timeout 3600s;
            keepalive_timeout 3600s;
            send_timeout 3600s;
            client_body_timeout 3600s;
            client_header_timeout 3600s;
            
            # Prevent nginx from closing connections
            proxy_ignore_client_abort on;
            proxy_socket_keepalive on;
            
            # TCP settings for real-time data
            tcp_nodelay on;
            tcp_nopush off;
        }
    }
}
EOF

# Stop any existing nginx
nginx -s stop 2>/dev/null || true
sleep 1

# Start nginx
echo "Starting nginx on port 8070, load balancing to 8080 and 8090..."
nginx -c /tmp/nginx-sse.conf
