python3 measure_performance.py --device cpu --n_env 1 --steps 288 &
python3 measure_performance.py --device cpu --n_env 10 --steps 288 &
python3 measure_performance.py --device cpu --n_env 100 --steps 288 &
python3 measure_performance.py --device cpu --n_env 1000 --steps 288 &

python3 measure_performance.py --device cuda --n_env 1 --steps 288 &
python3 measure_performance.py --device cuda --n_env 10 --steps 288 &
python3 measure_performance.py --device cuda --n_env 100 --steps 288 &
python3 measure_performance.py --device cuda --n_env 1000 --steps 288
wait

