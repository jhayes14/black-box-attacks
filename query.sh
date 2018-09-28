# NES
for i in {0..1000};
do
  echo $i;
  CUDA_VISIBLE_DEVICES=0 python main_.py --max-queries 1000000 --img-index $i --out-dir query_limited/ --target-class 0 --epsilon 0.05 --sensing-matrix NES --sigma 1e-2;
done 

# NES
for i in {0..1000};
do
  echo $i;
  CUDA_VISIBLE_DEVICES=0 python main_.py --max-queries 1000000 --img-index $i --out-dir query_limited/ --target-class 0 --epsilon 0.05 --sensing-matrix NES --sigma 1e-3;
done 

# NES
for i in {0..1000};
do
  echo $i;
  CUDA_VISIBLE_DEVICES=0 python main_.py --max-queries 1000000 --img-index $i --out-dir query_limited/ --target-class 0 --epsilon 0.05 --sensing-matrix NES --sigma 1e-4;
done 


# SPSA
for i in {0..1000};
do
  echo $i;
  CUDA_VISIBLE_DEVICES=0 python main_.py --max-queries 1000000 --img-index $i --out-dir query_limited/ --target-class 0 --epsilon 0.05 --sensing-matrix SPSA --sigma 1e-2;
done 

# SPSA
for i in {0..1000};
do
  echo $i;
  CUDA_VISIBLE_DEVICES=0 python main_.py --max-queries 1000000 --img-index $i --out-dir query_limited/ --target-class 0 --epsilon 0.05 --sensing-matrix SPSA --sigma 1e-3;
done 

# SPSA
for i in {0..1000};
do
  echo $i;
  CUDA_VISIBLE_DEVICES=0 python main_.py --max-queries 1000000 --img-index $i --out-dir query_limited/ --target-class 0 --epsilon 0.05 --sensing-matrix SPSA --sigma 1e-4;
done 

# SPSA1
for i in {0..1000};
do
  echo $i;
  CUDA_VISIBLE_DEVICES=1 python main_.py --max-queries 100000 --img-index $i --out-dir query_limited/ --target-class 0 --epsilon 0.05 --sensing-matrix SPSA1 --sigma 1e-2;
done 

# SPSA1
for i in {0..1000};
do
  echo $i;
  CUDA_VISIBLE_DEVICES=1 python main_.py --max-queries 1000000 --img-index $i --out-dir query_limited/ --target-class 0 --epsilon 0.05 --sensing-matrix SPSA1 --sigma 1e-3;
done 

# SPSA1
for i in {0..1000};
do
  echo $i;
  CUDA_VISIBLE_DEVICES=1 python main_.py --max-queries 1000000 --img-index $i --out-dir query_limited/ --target-class 0 --epsilon 0.05 --sensing-matrix SPSA1 --sigma 1e-4;
done 


# RDSA 
for i in {0..1000};
do
  echo $i;
  CUDA_VISIBLE_DEVICES=1 python main_.py --max-queries 1000000 --img-index $i --out-dir query_limited/ --target-class 0 --epsilon 0.05 --sensing-matrix RDSA --sigma 1e-2;
done 

# RDSA 
for i in {0..1000};
do
  echo $i;
  CUDA_VISIBLE_DEVICES=1 python main_.py --max-queries 1000000 --img-index $i --out-dir query_limited/ --target-class 0 --epsilon 0.05 --sensing-matrix RDSA --sigma 1e-3;
done 

# RDSA 
for i in {0..1000};
do
  echo $i;
  CUDA_VISIBLE_DEVICES=1 python main_.py --max-queries 1000000 --img-index $i --out-dir query_limited/ --target-class 0 --epsilon 0.05 --sensing-matrix RDSA --sigma 1e-4;
done 

