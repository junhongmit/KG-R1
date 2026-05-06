#!/bin/bash
set -e

echo "=========================================="
echo "  KG-R1 Data Setup - Knowledge Graph Data"
echo "=========================================="
echo ""

# Get the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "📁 Project root: $PROJECT_ROOT"
echo ""

# Change to project root for all operations
cd "$PROJECT_ROOT"

# Function to print section headers
print_section() {
    echo ""
    echo "=========================================="
    echo "  $1"
    echo "=========================================="
    echo ""
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check dependencies
print_section "Checking Dependencies"

dependencies=("python3" "git" "wget")
missing_deps=()

for dep in "${dependencies[@]}"; do
    if command_exists "$dep"; then
        echo "✅ $dep is installed"
    else
        echo "❌ $dep is not installed"
        missing_deps+=("$dep")
    fi
done

if [ ${#missing_deps[@]} -ne 0 ]; then
    echo ""
    echo "❌ Missing dependencies: ${missing_deps[*]}"
    echo "Please install them before continuing."
    exit 1
fi

echo ""
echo "✅ All dependencies are installed"

# Main menu
print_section "Data Setup Options"

echo "Available datasets to set up:"
echo ""
echo "1) ComplexWebQuestions (CWQ)"
echo "2) WebQuestionsSP (WebQSP)"
echo "3) Download Freebase KG data (required for all datasets)"
echo "4) Setup all datasets"
echo "5) Exit"
echo ""

read -p "Select an option (1-5): " choice

case $choice in
    1)
        print_section "Setting up ComplexWebQuestions (CWQ)"

        echo "ℹ️  CWQ dataset setup requires:"
        echo "   - Raw CWQ data in data_kg/CWQ/"
        echo "   - Freebase KG data (entities, relations)"
        echo ""

        if [ ! -d "data_kg/CWQ" ]; then
            echo "⚠️  Warning: data_kg/CWQ not found"
            echo "Please download CWQ dataset first from:"
            echo "https://www.dropbox.com/s/606rdq5c8bz3fct/final_data.zip"
            exit 1
        fi

        echo "Processing CWQ dataset..."
        python3 scripts/data_process_kg/cwq.py

        echo ""
        echo "Creating search-augmented training data..."
        python3 scripts/data_process_kg/cwq_search_augmented_initial_entities.py

        echo ""
        echo "✅ CWQ dataset setup complete!"
        echo "Training data available at: data_kg/cwq_search_augmented_initial_entities/"
        ;;

    2)
        print_section "Setting up WebQuestionsSP (WebQSP)"

        echo "ℹ️  WebQSP dataset setup requires:"
        echo "   - Raw WebQSP data in data_kg/webqsp/"
        echo "   - Freebase KG data (entities, relations)"
        echo ""

        if [ ! -d "data_kg/webqsp" ]; then
            echo "⚠️  Warning: data_kg/webqsp not found"
            echo "Please download WebQSP dataset first"
            exit 1
        fi

        echo "Processing WebQSP dataset..."
        python3 scripts/data_process_kg/webqsp.py

        echo ""
        echo "Creating search-augmented training data..."
        python3 scripts/data_process_kg/webqsp_search_augmented_initial_entities.py

        echo ""
        echo "✅ WebQSP dataset setup complete!"
        echo "Training data available at: data_kg/webqsp_search_augmented_initial_entities/"
        ;;

    3)
        print_section "Downloading Freebase KG Data"

        echo "This will download and process Freebase knowledge graph data."
        echo "This is a large download and may take significant time and disk space."
        echo ""
        read -p "Continue? (y/n): " confirm

        if [[ $confirm =~ ^[Yy]$ ]]; then
            python3 scripts/download_kg.py --save_path data_kg

            echo ""
            echo "✅ Freebase KG data downloaded!"
        else
            echo "Skipped Freebase KG download"
        fi
        ;;

    4)
        print_section "Setting up ALL datasets"

        echo "This will set up all available datasets."
        echo "⚠️  This requires:"
        echo "   - Significant disk space"
        echo "   - Significant time to download and process"
        echo "   - All raw dataset files to be present"
        echo ""
        read -p "Continue? (y/n): " confirm

        if [[ $confirm =~ ^[Yy]$ ]]; then
            # Download Freebase KG if needed
            if [ ! -f "data_kg/entities.txt" ]; then
                echo ""
                echo "Downloading Freebase KG data..."
                python3 scripts/download_kg.py --save_path data_kg
            fi

            # Setup CWQ
            if [ -d "data_kg/CWQ" ]; then
                echo ""
                print_section "Processing CWQ"
                python3 scripts/data_process_kg/cwq.py
                python3 scripts/data_process_kg/cwq_search_augmented_initial_entities.py
            fi

            # Setup WebQSP
            if [ -d "data_kg/webqsp" ]; then
                echo ""
                print_section "Processing WebQSP"
                python3 scripts/data_process_kg/webqsp.py
                python3 scripts/data_process_kg/webqsp_search_augmented_initial_entities.py
            fi

            echo ""
            print_section "Setup Complete!"
            echo "All datasets have been processed and are ready for training."
        else
            echo "Setup cancelled"
        fi
        ;;

    5)
        echo "Exiting..."
        exit 0
        ;;

    *)
        echo "❌ Invalid option"
        exit 1
        ;;
esac

print_section "Setup Summary"

echo "📋 Available training datasets:"
echo ""

datasets=(
    "data_kg/cwq_search_augmented_initial_entities"
    "data_kg/webqsp_search_augmented_initial_entities"
)

for dataset in "${datasets[@]}"; do
    if [ -d "$dataset" ]; then
        echo "✅ $dataset"
        if [ -f "$dataset/train.parquet" ]; then
            echo "   - train.parquet: $(du -h "$dataset/train.parquet" | cut -f1)"
        fi
        if [ -f "$dataset/test.parquet" ]; then
            echo "   - test.parquet: $(du -h "$dataset/test.parquet" | cut -f1)"
        fi
    else
        echo "⏭️  $dataset (not set up)"
    fi
done

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review the processed data in data_kg/"
echo "2. Update your training scripts to use the desired dataset"
echo "3. Start the KG retrieval server if needed"
echo "4. Run training with your chosen configuration"
echo ""
