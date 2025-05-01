#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to extract domain suffix from hostname
extract_domain_suffix() {
  local host="$1"
  # If it has at least one dot, extract everything after the first dot
  if [[ "$host" == *"."* ]]; then
    echo "$host" | sed -E 's/^[^.]+\.(.*)/\1/'
  else
    # If no dots, return the original
    echo "$host"
  fi
}

# Function to print section header
print_header() {
  echo -e "${BLUE}===================================================${NC}"
  echo -e "${BLUE}        $1        ${NC}"
  echo -e "${BLUE}===================================================${NC}"
}

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Check for required commands
print_header "Checking Requirements"
for cmd in kubectl jq; do
  if ! command_exists "$cmd"; then
    echo -e "${RED}Error: $cmd is not installed or not in PATH${NC}"
    exit 1
  fi
done

echo -e "${GREEN}✅ All required tools are available${NC}"

# Check if kubectl can access the cluster
echo -e "\n${YELLOW}Checking cluster connectivity...${NC}"
if ! kubectl get nodes &>/dev/null; then
  echo -e "${RED}Error: Cannot connect to Kubernetes cluster${NC}"
  echo -e "${YELLOW}Please make sure you're connected to your cluster and try again.${NC}"
  exit 1
fi

echo -e "${GREEN}✅ Connected to Kubernetes cluster${NC}"

# Initialize variables
ISTIO_INSTALLED=false
INGRESS_INSTALLED=false
CONFIG_TYPE=""
ISTIO_GATEWAY=""
ISTIO_HOST_SUFFIX=""
INGRESS_CLASS=""
INGRESS_HOST_SUFFIX=""

print_header "Detecting Network Components"

# Check for Istio installation
echo -e "${YELLOW}Checking for Istio...${NC}"
if kubectl get crd gateways.networking.istio.io &>/dev/null; then
  echo -e "${GREEN}✅ Istio is installed${NC}"
  ISTIO_INSTALLED=true

  # Get list of gateways
  echo -e "${YELLOW}Discovering Istio gateways...${NC}"
  GATEWAY_LIST=($(kubectl get gateway --all-namespaces -o jsonpath='{range .items[*]}{.metadata.namespace}/{.metadata.name}{"\n"}{end}' 2>/dev/null))

  if [ ${#GATEWAY_LIST[@]} -eq 0 ]; then
    echo -e "${YELLOW}No gateways found. Will use default suggestions.${NC}"
    GATEWAY_LIST=("istio-system/istio-ingressgateway")
  else
    echo -e "${GREEN}Found ${#GATEWAY_LIST[@]} gateway(s)${NC}"
  fi

  # Get host patterns from virtual services
  echo -e "${YELLOW}Discovering host patterns from VirtualServices...${NC}"
  HOST_PATTERNS=($(kubectl get virtualservices --all-namespaces -o json 2>/dev/null | \
    jq -r '.items[].spec.hosts[]?' | grep -v null | grep -v '\*' | \
    while read host; do extract_domain_suffix "$host"; done | sort | uniq))

  if [ ${#HOST_PATTERNS[@]} -eq 0 ]; then
    echo -e "${YELLOW}No host patterns found in VirtualServices.${NC}"
  else
    echo -e "${GREEN}Found ${#HOST_PATTERNS[@]} host pattern(s)${NC}"
  fi
else
  echo -e "${YELLOW}Istio is not installed${NC}"
fi

# Check for Ingress controllers
echo -e "\n${YELLOW}Checking for Ingress controllers...${NC}"
INGRESS_CLASSES=($(kubectl get ingressclass -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null))

if [ ${#INGRESS_CLASSES[@]} -gt 0 ]; then
  echo -e "${GREEN}✅ Ingress controller is installed${NC}"
  INGRESS_INSTALLED=true

  echo -e "${GREEN}Found ${#INGRESS_CLASSES[@]} IngressClass(es)${NC}"

  # Get default IngressClass if available
  DEFAULT_CLASS=$(kubectl get ingressclass -o json 2>/dev/null | \
    jq -r '.items[] | select(.metadata.annotations."ingressclass.kubernetes.io/is-default-class" == "true") | .metadata.name')

  if [ ! -z "$DEFAULT_CLASS" ]; then
    echo -e "${GREEN}Default IngressClass: ${DEFAULT_CLASS}${NC}"
  fi

  # Get host patterns from Ingress resources
  echo -e "${YELLOW}Discovering host patterns from Ingress resources...${NC}"
  INGRESS_HOST_PATTERNS=($(kubectl get ingress --all-namespaces -o json 2>/dev/null | \
    jq -r '.items[].spec.rules[].host?' | grep -v null | grep -v '\*' | \
    while read host; do extract_domain_suffix "$host"; done | sort | uniq))

  if [ ${#INGRESS_HOST_PATTERNS[@]} -eq 0 ]; then
    echo -e "${YELLOW}No host patterns found in Ingress resources.${NC}"
  else
    echo -e "${GREEN}Found ${#INGRESS_HOST_PATTERNS[@]} host pattern(s)${NC}"
  fi
else
  echo -e "${YELLOW}No Ingress controller found${NC}"
fi

# Interactive configuration
print_header "Network Configuration Wizard"

# If neither is installed, inform the user
if [ "$ISTIO_INSTALLED" = false ] && [ "$INGRESS_INSTALLED" = false ]; then
  echo -e "${RED}Neither Istio nor Ingress controller was detected in your cluster.${NC}"
  echo -e "${YELLOW}Please install one of them before running this wizard again.${NC}"
  exit 1
fi

# Ask which type to use if both are available
if [ "$ISTIO_INSTALLED" = true ] && [ "$INGRESS_INSTALLED" = true ]; then
  echo -e "${CYAN}Both Istio and Ingress controller are available in your cluster.${NC}"

  echo -e "${CYAN}Which would you like to use for your application?${NC}"
  PS3='Please choose (enter number): '

  options=("Istio" "Ingress Controller" "None (Skip network configuration)")

  select opt in "${options[@]}"; do
    case $opt in
      "Istio")
        CONFIG_TYPE="istio"
        break
        ;;
      "Ingress Controller")
        CONFIG_TYPE="ingress"
        break
        ;;
      "None (Skip network configuration)")
        CONFIG_TYPE="none"
        break
        ;;
      *)
        echo -e "${RED}Invalid option $REPLY${NC}"
        ;;
    esac
  done
elif [ "$ISTIO_INSTALLED" = true ]; then
  echo -e "${CYAN}Istio is available in your cluster.${NC}"
  read -p "Do you want to use Istio for your application? (y/n): " -r
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    CONFIG_TYPE="istio"
  else
    CONFIG_TYPE="none"
  fi
elif [ "$INGRESS_INSTALLED" = true ]; then
  echo -e "${CYAN}Ingress controller is available in your cluster.${NC}"

  read -p "Do you want to use the Ingress for your application? (y/n): " -r
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    CONFIG_TYPE="ingress"
  else
    CONFIG_TYPE="none"
  fi
fi

echo
# Configure based on selection
if [ "$CONFIG_TYPE" = "istio" ]; then
  echo -e "${CYAN}Configuring Istio settings...${NC}"

  # Select gateway
  if [ ${#GATEWAY_LIST[@]} -gt 0 ]; then
    echo -e "${CYAN}Which Istio gateway would you like to use?${NC}"
    PS3='Please choose a gateway (enter number): '
    select gateway in "${GATEWAY_LIST[@]}" "Enter custom gateway"; do
      if [ "$gateway" = "Enter custom gateway" ]; then
        read -p "Enter custom gateway (namespace/name): " ISTIO_GATEWAY
        break
      elif [ ! -z "$gateway" ]; then
        ISTIO_GATEWAY=$gateway
        break
      else
        echo -e "${RED}Invalid selection${NC}"
      fi
    done
  else
    read -p "Enter Istio gateway (namespace/name, default: istio-system/istio-ingressgateway): " ISTIO_GATEWAY
    if [ -z "$ISTIO_GATEWAY" ]; then
      ISTIO_GATEWAY="istio-system/istio-ingressgateway"
    fi
  fi

  # Select host suffix
  if [ ${#HOST_PATTERNS[@]} -gt 0 ]; then
    echo -e "${CYAN}Which host suffix would you like to use?${NC}"
    PS3='Please choose a host suffix (enter number): '
    select host_suffix in "${HOST_PATTERNS[@]}" "Enter custom host suffix"; do
      if [ "$host_suffix" = "Enter custom host suffix" ]; then
        read -p "Enter custom host suffix (e.g., example.com): " ISTIO_HOST_SUFFIX
        break
      elif [ ! -z "$host_suffix" ]; then
        ISTIO_HOST_SUFFIX=$host_suffix
        break
      else
        echo -e "${RED}Invalid selection${NC}"
      fi
    done
  else
    read -p "Enter host suffix (e.g., example.com): " ISTIO_HOST_SUFFIX
    if [ -z "$ISTIO_HOST_SUFFIX" ]; then
      read -p "Host suffix is required. Please enter a value: " ISTIO_HOST_SUFFIX
      while [ -z "$ISTIO_HOST_SUFFIX" ]; do
        read -p "Host suffix is required. Please enter a value: " ISTIO_HOST_SUFFIX
      done
    fi
  fi

elif [ "$CONFIG_TYPE" = "ingress" ]; then
  echo -e "${CYAN}Configuring Ingress settings...${NC}"

  # Select IngressClass
  if [ ${#INGRESS_CLASSES[@]} -gt 0 ] ; then
    echo -e "${CYAN}Which IngressClass would you like to use?${NC}"
    PS3='Please choose an IngressClass (enter number): '
    select ingress_class in "${INGRESS_CLASSES[@]}" "Enter custom IngressClass"; do
      if [ "$ingress_class" = "Enter custom IngressClass" ]; then
        read -p "Enter custom IngressClass: " INGRESS_CLASS
        break
      elif [ ! -z "$ingress_class" ]; then
        INGRESS_CLASS=$ingress_class
        break
      else
        echo -e "${RED}Invalid selection${NC}"
      fi
    done
  else
    read -p "Enter IngressClass (default: nginx): " INGRESS_CLASS
    if [ -z "$INGRESS_CLASS" ]; then
      INGRESS_CLASS="nginx"
    fi
  fi

  # Select host suffix
  if [ ${#INGRESS_HOST_PATTERNS[@]} -gt 0 ]; then
    echo -e "${CYAN}Which host suffix would you like to use?${NC}"
    PS3='Please choose a host suffix (enter number): '
    select host_suffix in "${INGRESS_HOST_PATTERNS[@]}" "Enter custom host suffix"; do
      if [ "$host_suffix" = "Enter custom host suffix" ]; then
        read -p "Enter custom host suffix (e.g., example.com): " INGRESS_HOST_SUFFIX
        break
      elif [ ! -z "$host_suffix" ]; then
        INGRESS_HOST_SUFFIX=$host_suffix
        break
      else
        echo -e "${RED}Invalid selection${NC}"
      fi
    done
  else
    read -p "Enter host suffix (e.g., example.com): " INGRESS_HOST_SUFFIX
    if [ -z "$INGRESS_HOST_SUFFIX" ]; then
      read -p "Host suffix is required. Please enter a value: " INGRESS_HOST_SUFFIX
      while [ -z "$INGRESS_HOST_SUFFIX" ]; do
        read -p "Host suffix is required. Please enter a value: " INGRESS_HOST_SUFFIX
      done
    fi
  fi
fi

# Generate values.yaml snippet
if [ "$CONFIG_TYPE" = "none" ]; then
  print_header "Configuration Summary"
  echo -e "${YELLOW}You've chosen not to configure network access.${NC}"
  echo -e "${YELLOW}Your application will not be exposed outside the cluster.${NC}"

elif [ "$CONFIG_TYPE" = "istio" ]; then
  print_header "Configuration Summary"
  echo -e "${YELLOW}Istio Configuration:${NC}"
  echo -e "Gateway: ${GREEN}$ISTIO_GATEWAY${NC}"
  echo -e "Host Suffix: ${GREEN}$ISTIO_HOST_SUFFIX${NC}"
  export ISTIO_ENABLED=true
  export INGRESS_ENABLED=false
  export ISTIO_GATEWAY=$ISTIO_GATEWAY
  export DYNAMO_INGRESS_SUFFIX=$ISTIO_HOST_SUFFIX

elif [ "$CONFIG_TYPE" = "ingress" ]; then
  print_header "Configuration Summary"
  echo -e "${YELLOW}Ingress Configuration:${NC}"
  echo -e "IngressClass: ${GREEN}$INGRESS_CLASS${NC}"
  echo -e "Host Suffix: ${GREEN}$INGRESS_HOST_SUFFIX${NC}"
  export INGRESS_ENABLED=true
  export ISTIO_ENABLED=false
  export INGRESS_CLASS=$INGRESS_CLASS
  export DYNAMO_INGRESS_SUFFIX=$INGRESS_HOST_SUFFIX
fi

print_header "Wizard Complete"
echo -e "${GREEN}Network configuration complete!${NC}"