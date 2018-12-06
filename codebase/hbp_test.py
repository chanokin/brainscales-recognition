import nmpi
from pprint import pprint
import time

client = nmpi.Client('garibaldi')
token = client.token
# pprint(client)
# pprint(token)

wafer = 33
hicann = 297
hw_config = {'WAFER_MODULE': wafer, 'HICANN': hicann, 
            'SOFTWARE_VERSION':'nmpm_software/current'}


job_id = client.submit_job(source='./portal_test.py',
                      platform=nmpi.BRAINSCALES,
                      collab_id=34089,
                      config=hw_config,
                      command="run.py --wafer {} --hicann {}".format(wafer, hicann),
            )

pprint(job_id)
# pprint(client.job_status(job_id))

job = client.get_job(job_id, with_log=True)
# pprint(job)

while client.job_status(job_id) != 'finished':
    time.sleep(1)

# pprint(client.job_status(job_id))

# pprint(job)

data = client.download_data(job, local_dir=".")

pprint(data)
