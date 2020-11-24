python mxe_parser.py --input ./data/bankAccountProduct-fullProduct.mxe  --output DeepLinker/bankAccountProduct-fullProduct  --number-of-node-features 1 --as-undirected False --generate-edge-symmetry False --add-node-labels True --add-info-firstline False --dublicate-node-features 256 --embeddings embeddings.txt
python mxe_parser.py --input ./data/emailProduct-fullProduct.mxe        --output DeepLinker/emailProduct-fullProduct        --number-of-node-features 1 --as-undirected False --generate-edge-symmetry False --add-node-labels True --add-info-firstline False --dublicate-node-features 256 --embeddings embeddings.txt
python mxe_parser.py --input ./data/svmProduct-fullProduct.mxe          --output DeepLinker/svmProduct-fullProduct          --number-of-node-features 1 --as-undirected False --generate-edge-symmetry False --add-node-labels True --add-info-firstline False --dublicate-node-features 256 --embeddings embeddings.txt
python mxe_parser.py --input ./data/sasProduct-fullProduct.mxe          --output DeepLinker/sasProduct-fullProduct          --number-of-node-features 1 --as-undirected False --generate-edge-symmetry False --add-node-labels True --add-info-firstline False --dublicate-node-features 256 --embeddings embeddings.txt

python mxe_parser.py --input ./data/bankAccountProduct-fullProduct.mxe  --output SEAL/bankAccountProduct-fullProduct        --number-of-node-features 1 --as-undirected False --generate-edge-symmetry False --add-node-labels False --add-info-firstline False --embeddings embeddings.txt
python mxe_parser.py --input ./data/emailProduct-fullProduct.mxe        --output SEAL/emailProduct-fullProduct              --number-of-node-features 1 --as-undirected False --generate-edge-symmetry False --add-node-labels False --add-info-firstline False --embeddings embeddings.txt
python mxe_parser.py --input ./data/svmProduct-fullProduct.mxe          --output SEAL/svmProduct-fullProduct                --number-of-node-features 1 --as-undirected False --generate-edge-symmetry False --add-node-labels False --add-info-firstline False --embeddings embeddings.txt
python mxe_parser.py --input ./data/sasProduct-fullProduct.mxe          --output SEAL/sasProduct-fullProduct                --number-of-node-features 1 --as-undirected False --generate-edge-symmetry False --add-node-labels False --add-info-firstline False --embeddings embeddings.txt
