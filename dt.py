import config as cfg
from classification import clf
from firewall import firewall_normal

def de_recognize(attack_commands):
    attack_types =[]
    for result in attack_commands["result"]:
        match result:
            case 0:
                attack_types.append('Normal')
            case 1:
                attack_types.append('NMRI')
            case 2:
                attack_types.append('CMRI')
            case 3:
                attack_types.append('MSCI')
            case 4:
                attack_types.append('MPCI')
            case 5:
                attack_types.append('MFCI')
            case 6:
                attack_types.append('DoS')
            case 7:
                attack_types.append('Reconnaissance')

    print(f"Attack Types: {attack_types}")

    locations = attack_commands["command_address"]
    print(f"Location: {locations}")


    return attack_types, locations

normal_commands = firewall_normal(cfg.TEST_DATASET)
attack_commands = clf(normal_commands)[1]
attack_types = de_recognize(attack_commands)