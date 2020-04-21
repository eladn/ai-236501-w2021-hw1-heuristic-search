from framework import *
from problems import *

import numpy as np
from typing import *
from warnings import warn


PERSON_FULL_NAMES = """
Doris Sollers
Maryjane Dickert
Mindy Drennen
Kyle Turck
Lannie Hunkins
Krysta Valentine
Ethan Mulholland
Cleora Alaniz
Leia Urquhart
Natalia Templin
Jasmine Denson
Quinton Salais
Raven Woolum
Gayle Schild
Hobert Venable
Anisha Decker
Johnny Fierro
Tess Bish
Christopher Dickinson
Kurt Dockstader
Hana Hockman
Stefani Kreutzer
Nettie Eggebrecht
Leif Haug
Meda Geisler
Cyril Boyland
Gussie Foran
Margorie Bouldin
Lizzette Kohn
Hedy Moser
Kori Satcher
Kirstie Demyan
Lyla Motto
Sidney Dunmire
Arron Mungin
Willie Luellen
Max Neri
Tegan Storms
Stacia Landsman
Marlen Ehle
Julia Hadlock
Pierre Lowman
Veronique Katz
Debi Basta
Edie Echevarria
Rhett Klopfenstein
Felicitas Cardone
Elicia Klemme
Joaquina Olsson
Jeane Ravencraft
""".strip().split('\n')
PERSON_FULL_NAMES = [name.strip() for name in PERSON_FULL_NAMES if name.strip()]
assert all(len(name.split(' ')) == 2 for name in PERSON_FULL_NAMES)
PERSON_FIRST_NAMES = list(set(name.split()[0].strip() for name in PERSON_FULL_NAMES))
PERSON_LAST_NAMES = list(set(name.split()[1].strip() for name in PERSON_FULL_NAMES) - set(PERSON_FIRST_NAMES))


def get_rand_input_junctions(
        roads: StreetsMap,
        nr_junctions: int,
        random_state: np.random.RandomState,
        limit_to_radius: Optional[float] = None) -> List[Junction]:
    all_junction_indices = list(roads.keys())
    if limit_to_radius:
        center_junction_idx = random_state.choice(all_junction_indices, 1)[0]
        center_junction = roads[center_junction_idx]
        all_junction_indices = [
            junction_idx
            for junction_idx, junction in roads.items()
            if center_junction.calc_air_distance_from(junction) <= limit_to_radius
        ]

    assert nr_junctions < 0.2 * len(all_junction_indices)
    all_junction_indices = np.array(all_junction_indices)
    chosen_junction_idxs = random_state.choice(all_junction_indices, nr_junctions, replace=False)
    assert all(idx in roads for idx in chosen_junction_idxs)
    assert len(set(chosen_junction_idxs)) == nr_junctions
    return [roads[idx] for idx in chosen_junction_idxs]


def generate_MDA_problem_input(
        roads: StreetsMap,
        input_name: str,
        choosing_junctions_seed: int = 0x5739574,
        limit_to_radius: int = 6000,
        nr_reported_apartments: int = 5,
        nr_laboratories: int = 5,
        ambulance_taken_tests_storage_capacity: int = 7,
        initial_nr_matoshim_on_ambulance: int = 2,
        nr_roommates_options: Tuple[int, ...] = (2, 3, 4, 5),
        nr_roommates_probabilities: Tuple[float, ...] = (0.2, 0.3, 0.2, 0.3),
        nr_free_matoshim_in_lab_options: Tuple[int, ...] = (4, 5, 6, 7, 8),
        nr_free_matoshim_in_lab_probabilities: Tuple[float, ...] = (0.2, 0.3, 0.2, 0.2, 0.1)) -> MDAProblemInput:

    random_state = np.random.RandomState(choosing_junctions_seed)
    all_sampled_junctions = get_rand_input_junctions(
        roads=roads,
        nr_junctions=nr_reported_apartments + nr_laboratories + 1,
        random_state=random_state, limit_to_radius=limit_to_radius
    )

    reported_apartments_junctions = all_sampled_junctions[0:nr_reported_apartments]
    assert len(reported_apartments_junctions) == nr_reported_apartments
    laboratories_junctions = all_sampled_junctions[nr_reported_apartments:nr_reported_apartments + nr_laboratories]
    assert len(laboratories_junctions) == nr_laboratories
    initial_truck_location = all_sampled_junctions[nr_reported_apartments + nr_laboratories]

    names = random_state.permutation(PERSON_FULL_NAMES)
    if nr_reported_apartments > len(names):
        warn(f'More reported apartments to generate ({nr_reported_apartments}) than names ({len(names)}).')

    last_names_permutation = random_state.permutation(PERSON_LAST_NAMES)
    labs_names = [f'{name1}-{name2}'
                  for pair_idx, (name1, name2) in enumerate(zip(last_names_permutation, last_names_permutation[1:]))
                  if pair_idx % 2 == 0]
    if nr_laboratories > len(labs_names):
        warn(f'More laboratories to generate ({nr_laboratories}) than last names pairs ({len(labs_names)}).')

    return MDAProblemInput(
        input_name=input_name,
        reported_apartments=tuple(
            ApartmentWithSymptomsReport(
                report_id=report_idx + 1,
                reporter_name=names[report_idx % len(names)],
                location=junction,
                nr_roommates=random_state.choice(nr_roommates_options, p=nr_roommates_probabilities)
            )
            for report_idx, junction
            in enumerate(reported_apartments_junctions)),
        ambulance=Ambulance(
            initial_nr_matoshim=initial_nr_matoshim_on_ambulance,
            taken_tests_storage_capacity=ambulance_taken_tests_storage_capacity,
            initial_location=initial_truck_location),
        laboratories=tuple(
            Laboratory(
                lab_id=lab_id, name=labs_names[lab_id % len(labs_names)], location=junction,
                max_nr_matoshim=random_state.choice(
                    nr_free_matoshim_in_lab_options, p=nr_free_matoshim_in_lab_probabilities))
            for lab_id, junction in enumerate(laboratories_junctions)))


def generate_MDA_problem_inputs_files(roads: StreetsMap):
    inputs = [
        generate_MDA_problem_input(
            roads,
            input_name='small_MDA',
            choosing_junctions_seed=0x5739574,
            limit_to_radius=6000,
            nr_reported_apartments=5,
            ambulance_taken_tests_storage_capacity=9,
            nr_roommates_options=(2, 3, 4, 5),
            nr_roommates_probabilities=(0.2, 0.3, 0.2, 0.3)),
        generate_MDA_problem_input(
            roads,
            input_name='moderate_MDA',
            choosing_junctions_seed=0x5739574,
            limit_to_radius=6000,
            nr_reported_apartments=8,
            nr_laboratories=4,
            ambulance_taken_tests_storage_capacity=6,
            initial_nr_matoshim_on_ambulance=3,
            nr_roommates_options=(1, 2, 3, 4),
            nr_roommates_probabilities=(0.2, 0.3, 0.3, 0.2)),
        generate_MDA_problem_input(
            roads,
            input_name='big_MDA',
            choosing_junctions_seed=0x5739574,
            limit_to_radius=6000,
            nr_reported_apartments=15,
            nr_laboratories=5,
            ambulance_taken_tests_storage_capacity=5,
            nr_roommates_options=(2, 3, 4, 5),
            nr_roommates_probabilities=(0.2, 0.3, 0.2, 0.3)),
        generate_MDA_problem_input(
            roads,
            input_name='test_MDA_small',
            choosing_junctions_seed=0x2484424,
            limit_to_radius=6000,
            nr_reported_apartments=5,
            ambulance_taken_tests_storage_capacity=9,
            nr_roommates_options=(2, 3, 4, 5),
            nr_roommates_probabilities=(0.2, 0.3, 0.2, 0.3)),
        generate_MDA_problem_input(
            roads,
            input_name='test_MDA_medium',
            choosing_junctions_seed=0x2484424,
            limit_to_radius=6000,
            nr_reported_apartments=8,
            ambulance_taken_tests_storage_capacity=9,
            nr_roommates_options=(2, 3, 4, 5),
            nr_roommates_probabilities=(0.2, 0.3, 0.2, 0.3)),
    ]
    for problem_input in inputs:
        problem_input.store_to_file(problem_input.input_name + '.in')


def update_streets_map():
    print('Generating new streets-map with toll roads and max speed per road ..')
    streets_map = StreetsMap.load_from_csv(Consts.get_data_file_path('tlv_original.csv'))
    print(f'#links: {sum(1 for _ in streets_map.iterlinks())} (before remove dangling links)')
    streets_map.remove_dangling_links()
    print(f'#links: {sum(1 for _ in streets_map.iterlinks())} (after remove dangling links, before remove zero dist links)')
    streets_map.remove_zero_distance_links()
    print(f'#links: {sum(1 for _ in streets_map.iterlinks())} (after remove dangling links, after remove zero dist links)')
    streets_map.update_link_distances_to_air_distance()
    streets_map.set_links_max_speed_and_is_toll()
    streets_map.set_incoming_links()
    streets_map.write_to_csv(Consts.get_data_file_path('tlv_streets_map.csv'))
    print('Done generating the new streets-map.')


if __name__ == '__main__':
    # update_streets_map()
    # exit()
    streets_map = StreetsMap.load_from_csv(Consts.get_data_file_path('tlv_streets_map.csv'))
    generate_MDA_problem_inputs_files(streets_map)
