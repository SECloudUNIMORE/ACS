# DAGA Dataset
This folder contains the [clean](clean) and [infected](infected) dataset used for training and testing DAGA.
The files of the dataset contains CAN data frames and metadata and are organized in a ```CSV``` format (compressed in zips) with the following columns:
 - ```timestamp```: the relative timestamp of the message;
 - ```CAN_ID```: hexadecimal value of the CAN message ID;
 - ```DLC```: data length code, expressed in bytes;
 - ```PAYLOAD_HEX```: hexadecimal representation of the payload field of the CAN data frame;
 - ```PAYLOAD_BIN```: binary representation of the payload field of the CAN data frame;
 - ```ANOMALY```: flag to distinguish between legit and malicious messages, used in the detection phase for performance evaluation. 

## Clean dataset
The clean dataset is collected from an unmodified, licensed 2016 Volvo V40 Kinetic model. In the [clean dataset folder](clean) there are **7** different can traces, each one containing the data gathered from a different driving scenario. The values of the ```ANOMALY``` field in the clean log files are fixed at ```FALSE```.

## Infected dataset
The infected dataset is generated in a laboratory setup by replicating the attacks composing the threat model considered in our work. In the [infected dataset folder](infected) there are **3** different folders:
 - ```DenialOfService```: Denial-of-Service on the CAN communication. This folder contains **7** attack traces;
 - ```OrderedSequenceReplay```: Replay of an ordered sequence of CAN messages. This folder contains **63** attack traces, composed by injecting a sequence of length *n* on a legit CAN trace. The file names are encoded as ```n_[X]_[TRACE]```, where ```[X]``` is the length of the injected sequence *n* and ```[TRACE]``` is the name of the clean log file used for generating the attack;
 - ```SingleIDReplay```: Replay of a single CAN message. This folder contains **28** attack traces, composed by injecting a single message with a selected ID on a legit CAN trace. The file names are encoded as ```ID_[mID]_[TRACE]```, where ```[mID]``` is the hexadecimal representation of the injected CAN ID and ```[TRACE]``` is the name of the clean log file used for generating the attack;
 - ```UnorderedSequenceReplay```: Replay of an unordered sequence of CAN messages. This folder contains **63** attack traces, composed by injecting a sequence of length *n* on a legit CAN trace. The file names are encoded as the ones found in the ```OrderedSequenceReplay``` folder.

For a detailed explanation of the dataset, please refer to the original paper.
