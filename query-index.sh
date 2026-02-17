#!/bin/bash
#
# Simple OpenSearch Index Query Tool
# Query any OpenSearch index for recent entries
#

set -u

# Colors
C_INFO="\033[0;36m"
C_SUCCESS="\033[0;32m"
C_ERROR="\033[0;31m"
C_YELLOW="\033[0;33m"
C_RESET="\033[0m"

# Common index URLs (for quick selection)
# Note: Query endpoint is different from write endpoint
# Write: http://gpuwa.nvidia.com/dataflow2/...
# Query: https://gpuwa.nvidia.com/opensearch/df-...*
declare -A KNOWN_INDEXES
KNOWN_INDEXES=(
    ["workflows"]="https://gpuwa.nvidia.com/opensearch/df-swdl-triton-ops-gh-workflows*"
    ["jobs"]="https://gpuwa.nvidia.com/opensearch/df-swdl-triton-ops-gh-jobs*"
    ["steps"]="https://gpuwa.nvidia.com/opensearch/df-swdl-triton-ops-gh-steps*"
    ["containers"]="https://gpuwa.nvidia.com/opensearch/df-swdl-triton-ops-gh-containers*"
    ["stages"]="https://gpuwa.nvidia.com/opensearch/df-swdl-triton-ops-gh-stages*"
    ["layers"]="https://gpuwa.nvidia.com/opensearch/df-swdl-triton-ops-gh-layers*"
    ["tests"]="https://gpuwa.nvidia.com/opensearch/df-swdl-triton-ops-gh-tests*"
)

# Logging
log_info() { echo -e "${C_INFO}[INFO]${C_RESET} $1"; }
log_success() { echo -e "${C_SUCCESS}[SUCCESS]${C_RESET} $1"; }
log_error() { echo -e "${C_ERROR}[ERROR]${C_RESET} $1"; }
log_warn() { echo -e "${C_YELLOW}[WARN]${C_RESET} $1"; }

# Calculate ISO timestamp for hours back
get_timestamp_hours_ago() {
    local hours=$1
    if command -v date >/dev/null 2>&1; then
        # Check if GNU date or BSD date
        if date --version >/dev/null 2>&1; then
            # GNU date
            date -u -d "$hours hours ago" +"%Y-%m-%dT%H:%M:%S.000Z" 2>/dev/null || echo ""
        else
            # BSD date (macOS)
            date -u -v-${hours}H +"%Y-%m-%dT%H:%M:%S.000Z" 2>/dev/null || echo ""
        fi
    else
        echo ""
    fi
}

# Check dependencies
check_dependencies() {
    local missing=()
    
    if ! command -v curl >/dev/null 2>&1; then
        missing+=("curl")
    fi
    
    if ! command -v jq >/dev/null 2>&1; then
        missing+=("jq")
    fi
    
    if [ ${#missing[@]} -gt 0 ]; then
        log_error "Missing required dependencies: ${missing[*]}"
        log_error "Install with: yum install ${missing[*]}"
        exit 1
    fi
}

# Query OpenSearch index
query_index() {
    local index_url=$1
    local hours=$2
    local size=$3
    local use_time_filter=${4:-true}
    
    # Determine the search URL
    local search_url
    if [[ "$index_url" == *"/_search" ]]; then
        # Already has /_search suffix
        search_url="$index_url"
    elif [[ "$index_url" == *"/_doc" ]]; then
        # Has /_doc suffix, replace with /_search
        search_url="${index_url/_doc/_search}"
    elif [[ "$index_url" == */posting ]]; then
        # Has /posting suffix (write endpoint), replace with /_search
        search_url="${index_url/\/posting/\/_search}"
    else
        # No suffix, append /_search
        search_url="${index_url}/_search"
    fi
    
    # Build query
    local query
    if [ "$use_time_filter" == "true" ]; then
        local start_time=$(get_timestamp_hours_ago "$hours")
        local now=$(date -u +"%Y-%m-%dT%H:%M:%S.000Z")
        
        if [ -n "$start_time" ]; then
            # Query with time range
            query=$(cat <<EOF
{
  "size": $size,
  "sort": [{"@timestamp": {"order": "desc"}}],
  "query": {
    "range": {
      "@timestamp": {
        "gte": "$start_time",
        "lte": "$now"
      }
    }
  }
}
EOF
)
        else
            # Fallback if date calculation fails
            query=$(cat <<EOF
{
  "size": $size,
  "sort": [{"@timestamp": {"order": "desc"}}]
}
EOF
)
            log_warn "Could not calculate time range, querying for most recent $size entries"
            use_time_filter="false"
        fi
    else
        # Query without time range (just get recent)
        query=$(cat <<EOF
{
  "size": $size,
  "sort": [{"@timestamp": {"order": "desc"}}]
}
EOF
)
    fi
    
    log_info "Querying: $search_url"
    if [ "$use_time_filter" == "true" ]; then
        log_info "Time range: Last $hours hours"
        local start_time=$(get_timestamp_hours_ago "$hours")
        local now=$(date -u +"%Y-%m-%dT%H:%M:%S.000Z")
        if [ -n "$start_time" ]; then
            log_info "Start time: $start_time"
            log_info "End time: $now"
        fi
    else
        log_info "No time filter - getting most recent entries"
    fi
    log_info "Max results: $size"
    
    # Show query if DEBUG is set
    if [ "${DEBUG:-false}" == "true" ]; then
        echo ""
        echo "DEBUG: Query being sent:"
        echo "$query" | jq '.'
        echo ""
    fi
    
    echo ""
    
    # Execute query
    local result=$(curl -s -X POST "$search_url" \
        -H "Content-Type: application/json" \
        -d "$query")
    
    # Check if query was successful
    if [ -z "$result" ] || [ "$result" == "null" ]; then
        log_error "Failed to query OpenSearch or no response received"
        return 1
    fi
    
    # Show raw response if DEBUG is set
    if [ "${DEBUG:-false}" == "true" ]; then
        echo "DEBUG: Raw response:"
        echo "$result" | jq '.' 2>/dev/null || echo "$result"
        echo ""
    fi
    
    # Check for error in response
    local error=$(echo "$result" | jq -r '.error // empty' 2>/dev/null)
    if [ -n "$error" ]; then
        log_error "OpenSearch returned an error:"
        echo "$result" | jq -r '.error' 2>/dev/null || echo "$error"
        return 1
    fi
    
    # Get total hits
    local total=$(echo "$result" | jq -r '.hits.total.value // .hits.total // 0' 2>/dev/null || echo "0")
    
    echo "=========================================="
    echo "✅ Found $total total entries, showing top $size:"
    echo "=========================================="
    echo ""
    
    if [ "$total" == "0" ]; then
        log_warn "No entries found in the specified time range"
        return 0
    fi
    
    # Display results in a readable format
    echo "$result" | jq -r '.hits.hits[] | 
        "📄 Entry #" + (._index // "unknown") + "/" + (._id // "unknown") +
        "\n   Timestamp: " + (.["_source"]["@timestamp"] // "N/A") +
        "\n   Source Fields:" +
        "\n" +
        ((.["_source"] | to_entries | 
            map(
                if .key != "@timestamp" then
                    "      " + .key + ": " + (.value | tostring)
                else
                    empty
                end
            ) | join("\n")
        ) // "      (no fields)") +
        "\n" +
        "------------------------------------------"
    ' 2>/dev/null || {
        log_error "Failed to parse results"
        echo "$result" | jq '.' 2>/dev/null || echo "$result"
        return 1
    }
}

# Show list of known indexes
show_indexes() {
    echo ""
    echo "Known Indexes:"
    echo "=========================================="
    for key in "${!KNOWN_INDEXES[@]}"; do
        echo "  $key -> ${KNOWN_INDEXES[$key]}"
    done | sort
    echo "=========================================="
    echo ""
}

# Interactive mode - prompt user for index
interactive_mode() {
    show_indexes
    
    echo "Enter index to query:"
    echo "  - Type a shortcut name (e.g., 'workflows', 'tests')"
    echo "  - Or enter a full URL (e.g., 'http://host/index')"
    echo ""
    read -p "Index: " index_input
    
    if [ -z "$index_input" ]; then
        log_error "No index specified"
        exit 1
    fi
    
    # Check if it's a known shortcut
    local index_url=""
    if [ -n "${KNOWN_INDEXES[$index_input]:-}" ]; then
        index_url="${KNOWN_INDEXES[$index_input]}"
        log_info "Using known index: $index_input"
    else
        index_url="$index_input"
        log_info "Using custom index URL"
    fi
    
    # Ask for hours back
    read -p "Hours back [default: 24]: " hours_input
    local hours=${hours_input:-24}
    
    # Ask for max results
    read -p "Max results [default: 10]: " size_input
    local size=${size_input:-10}
    
    echo ""
    query_index "$index_url" "$hours" "$size"
}

# Usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] [INDEX]

Simple tool to query any OpenSearch index for recent entries

OPTIONS:
    --index URL             Index URL or shortcut name (required if not interactive)
    --hours N               Hours to look back (default: 24)
    --size N                Max results to return (default: 10)
    --no-time-filter        Don't filter by time, just get most recent entries
    --list                  Show list of known indexes
    --interactive, -i       Interactive mode (prompts for input)
    --debug                 Enable debug output
    -h, --help              Show this help

INDEX SHORTCUTS:
$(for key in "${!KNOWN_INDEXES[@]}"; do echo "    $key"; done | sort)

EXAMPLES:
    # Interactive mode (prompts for input)
    $0 --interactive
    $0 -i

    # Query workflows index
    $0 --index workflows --hours 48 --size 10

    # Query tests index
    $0 --index tests --hours 12 --size 20

    # Query custom index URL
    $0 --index http://custom.com/my-index --hours 24

    # List known indexes
    $0 --list

REQUIREMENTS:
    - curl (for HTTP requests)
    - jq (for JSON parsing)

EOF
    exit 0
}

# Main function
main() {
    check_dependencies
    
    # Default values
    local index_url=""
    local hours=24
    local size=10
    local interactive=false
    local use_time_filter=true
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --index)
                local index_input="$2"
                # Check if it's a shortcut
                if [ -n "${KNOWN_INDEXES[$index_input]:-}" ]; then
                    index_url="${KNOWN_INDEXES[$index_input]}"
                else
                    index_url="$index_input"
                fi
                shift 2
                ;;
            --hours)
                hours="$2"
                shift 2
                ;;
            --size)
                size="$2"
                shift 2
                ;;
            --no-time-filter)
                use_time_filter=false
                shift
                ;;
            --debug)
                export DEBUG=true
                shift
                ;;
            --list)
                show_indexes
                exit 0
                ;;
            --interactive|-i)
                interactive=true
                shift
                ;;
            -h|--help)
                usage
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                ;;
        esac
    done
    
    # Run interactive mode or regular mode
    if [ "$interactive" == "true" ]; then
        interactive_mode
    else
        if [ -z "$index_url" ]; then
            log_error "No index specified. Use --index or --interactive"
            echo ""
            usage
        fi
        
        query_index "$index_url" "$hours" "$size" "$use_time_filter"
    fi
    
    log_success "Query completed"
}

# Run main
main "$@"

