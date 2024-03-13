
# +
from config import *
import subprocess
import time
#import Arduino_read


########################## Run a Config on board ############################
#def Run_Graph(ALL_Freqs, run_command, myoutput, blocking=True):
### Separated prepare and run to solve the challenge of when starting the power profile
def Run_Graph(ALL_Freqs, prepare_command, output_filename, blocking=True,Power_monitoring=None):
    #This will prepare everything and write run command in run_command.sh
    with open(output_filename+"_log_", 'w') as myoutput_log:
        print(f'prepare command is:{prepare_command}')
        #p = subprocess.Popen(prepare_command.split(),shell=True,text=True)
        p = subprocess.Popen(prepare_command.split(), stdout=myoutput_log, stderr=myoutput_log, stdin=subprocess.PIPE, text=True)
        p.wait()
    print("\n*********************************************\n\
    Preparation Finished\n***********************************************\n\n")
    time.sleep(3)
    if p.returncode == 0:
        print("Subprocess was successful")
        #return -1
    else:
        print("Subprocess failed with return code:", p.returncode)
        return -1
    #input("h\n")
    run_command=f"{cnn_dir}/run_command.sh"
    Power_monitoring.start()
    with open(output_filename, 'w') as myoutput:
        print(f'run command is:{run_command}')
        p = subprocess.Popen(run_command.split(),stdout=myoutput,stderr=myoutput, stdin=subprocess.PIPE, text=True)
        #time.sleep(50)
        time.sleep(5)
        for Freqs in ALL_Freqs:             
            '''while p.poll() is None:
                # check if the subprocess is ready to accept input
                _, wlist, xlist = select.select([], [p.stdin], [p.stdin], 1)
                if wlist:  # Ready for writing
                    print("Ready to write freq to stdin of the process")
                    break
                if xlist:  # Exceptional condition
                    raise Exception("Exceptional condition on subprocess stdin")'''
            p.stdin.write(f'{Freqs}\n')
            p.stdin.flush()

            '''while p.poll() is None:
                # check if the subprocess is ready to accept input
                rlist, _, _ = select.select([p.stdin], [], [], 1)
                if rlist:
                    break'''

            time.sleep(12)

        p.stdin.write("end\n")
        p.stdin.flush()
        if blocking:
            p.wait()

# # +

# +

        

    
def Run_Graph_1(ALL_Freqs, run_command, output_filename, blocking=True,Power_monitoring=None):
    with open(output_filename, 'w') as myoutput:
        print(run_command)
        p = subprocess.Popen(run_command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)

        in_text_power="setup finished"
        in_text="Please Enter the desired Frequency setttings:"
        freq_index = 0
        while freq_index < len(ALL_Freqs) or p.poll() is None:
            # Use select to wait for output
            readable, _, _ = select.select([p.stdout, p.stderr], [], [], 1)
            for stream in readable:
                line = stream.readline()
                if line:
                    print('Output:', line, end='')
                    myoutput.write(line)
                    myoutput.flush()
                    if in_text_power in line:
                        Power_monitoring.start()
                        print('start pm in Run_graph function\n')
                        time.sleep(4)
                    if in_text in line and freq_index < len(ALL_Freqs):
                        p.stdin.write(f'{ALL_Freqs[freq_index]}\n')
                        p.stdin.flush()
                        freq_index += 1

            # Adjust as needed for your use case
            #time.sleep(1)

        p.stdin.write("end\n")
        p.stdin.flush()
        if blocking:
            p.wait()
    
    
def enqueue_output(out, queue, file):
    for line in iter(out.readline, ''):
        file.write(line)
        file.flush()
        if(queue):
            queue.put(line)
    print("\n\n\n\n\n\n\n\n\n\n\n\nTAmam\n\n\n\n\n\n")
    out.close()
        
def Run_Graph_2(ALL_Freqs, run_command, output_filename, blocking=True,Power_monitoring=None):
    with open(output_filename, 'w') as myoutput:
        with open(output_filename+'_log', 'w') as myoutput_log:
            # Start the subprocess with stdout and stderr redirected to pipes
            p = subprocess.Popen(run_command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)

            # Queues for stdout and stderr
            stdout_queue = queue.Queue()
            #stderr_queue = queue.Queue()

            # Threads for stdout and stderr
            stdout_thread = threading.Thread(target=enqueue_output, args=(p.stdout, stdout_queue, myoutput))
            #stderr_thread = threading.Thread(target=enqueue_output, args=(p.stderr, None, myoutput_log))
            stdout_thread.daemon = True
            #stderr_thread.daemon = True
            stdout_thread.start()
            #stderr_thread.start()

            freq_index = 0
            in_text_power="setup finished"
            in_text="Please Enter the desired Frequency setttings:"
            while freq_index < len(ALL_Freqs):
                # Check stdout and stderr
                for q in [stdout_queue]:#, stderr_queue]:
                    try:
                        line = q.get_nowait()
                    except queue.Empty:
                        continue  # No output yet, keep checking
                    else:
                        if in_text_power in line:
                            Power_monitoring.start()
                            print("Starting power monitoring in run_graph func\n")
                            time.sleep(2)
                        print('Output:', line)
                        
                        if in_text in line:
                            if freq_index < len(ALL_Freqs):
                                p.stdin.write(f'{ALL_Freqs[freq_index]}\n')
                                p.stdin.flush()
                                freq_index += 1
                #time.sleep(0.0001)  # Adjust sleep time as needed

            p.stdin.write("end\n")
            p.stdin.flush()
            if blocking:
                p.wait()

            stdout_thread.join()
            #stderr_thread.join()



