"""
Standardised output path generation for clustering results.

Generates a directory hierarchy and filenames for centroids, labels,
and metadata JSON files based on experiment tag, k, deletion size,
and deletion position.
"""
# generate_filename.py
import os
import sys
def generate_filename(parent_dir, tag, num_clusters, deletion_size, deletion_position):
    base_dir_template = os.path.join(parent_dir, tag, f'delSize_{deletion_size}', f'k_{num_clusters}')
    filenames = {}
    for file_type, extension in [('centroids', 'npy'), ('labels', 'npy'), ('meta', 'json')]:
        if file_type in ['centroids', 'labels']:
            base_dir = os.path.join(base_dir_template, file_type)
        else:
            base_dir = base_dir_template
        filename = f"{tag}_{file_type}_k{num_clusters}_delSize{deletion_size}_delPosP{deletion_position}.{extension}"
        filenames[file_type] = os.path.join(base_dir, filename)
    return filenames

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python generate_filename.py parent_dir tag num_clusters deletion_size deletion_position")
        sys.exit(1)
    _, parent_dir, tag, num_clusters, deletion_size, deletion_position = sys.argv
    filenames = generate_filename(parent_dir, tag, int(num_clusters), int(deletion_size), int(deletion_position))
    print(filenames['meta'])  
