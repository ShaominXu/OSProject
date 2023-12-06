import config as cfg
from classification import clf
from firewall import firewall_normal

def emp(nomral_commands):

    df = nomral_commands
    # Define a condition for normal commands (measurement is not zero)
    normal_commands = df[df['measurement'] != 0]

    # Define a condition for attack commands (measurement is zero)
    attack_commands = df[df['measurement'] == 0]

    # Print or use the normal_commands and attack_commands dataframes as needed
    print("Normal Commands:")
    print(normal_commands)

    print("\nAttack Commands:")
    print(attack_commands)

    return normal_commands

if __name__ == '__main__':
    normal_commands = firewall_normal(cfg.TEST_DATASET)
    normal_commands = clf(normal_commands)[0]
    normal_commands = emp(normal_commands)