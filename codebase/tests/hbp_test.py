import nmpi
from pprint import pprint
import time

client = nmpi.Client('garibaldi')
token = client.token
# pprint(client)
# pprint(token)

GITHUB = bool(1)
wait_time_s = 10
wafer = 33
hicann = 297
hw_config = {
    'WAFER_MODULE': wafer, 
    # 'HICANN': hicann, 
    # 'SOFTWARE_VERSION':'nmpm_software/current'
}


if GITHUB:
    job_id = client.submit_job(
        source='https://github.com/chanokin/brainscales-recognition.git',
        platform=nmpi.BRAINSCALES,
        collab_id=34089,
        config=hw_config,
        command="codebase/tests/portal_test.py --wafer {} --hicann {}".format(wafer, hicann),
    )
else:
    job_id = client.submit_job(
        source="portal_test.py",
        platform=nmpi.BRAINSCALES,
        collab_id=34089,
        config=hw_config,
    )

pprint(job_id)
# pprint(client.job_status(job_id))

job = client.get_job(job_id, with_log=True)
# pprint(job)

while client.job_status(job_id) != 'finished':
    print("Not finished, waiting {} seconds".format(wait_time_s))
    time.sleep(wait_time_s)

time.sleep(wait_time_s)

pprint(client.job_status(job_id))

pprint(job)

data = client.download_data(job, local_dir="./job_{}".format(job_id))

pprint(data)
