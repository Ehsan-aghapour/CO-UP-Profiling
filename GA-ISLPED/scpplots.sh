
mkdir -p $1/plots/
mkdir -p $1/csvs/
#copy plots from gpu server
scp 145.100.131.33:/home/ehsan/CO-UP-Profiling/GA-ISLPED/*.jpg /home/ehsan/UvA/ARMCL/Rock-Pi/CO-UP-Profiling/GA-ISLPED/$1/plots/
scp 145.100.131.33:/home/ehsan/CO-UP-Profiling/GA-ISLPED/*.csv /home/ehsan/UvA/ARMCL/Rock-Pi/CO-UP-Profiling/GA-ISLPED/$1/csvs/
#scp 145.100.131.33:/home/ehsan/CO-UP-Profiling/GA-ISLPED/*.pkl /home/ehsan/UvA/ARMCL/Rock-Pi/CO-UP-Profiling/GA-ISLPED/1/pkls/
