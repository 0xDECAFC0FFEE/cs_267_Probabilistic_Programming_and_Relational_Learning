import os
import pandas as pd
import random
import numpy as np
from utils import *
from itertools import product
import re
from ast import literal_eval
from pathlib import Path
import pickle
import bz2

validation_set_source = "test" # can be one of ["test", ["phase", "i"]]

def filename_from_args(dataset_path, traintest, phase, filetype, extension):
    """gets folder structure filename from arguments

    Arguments:
        dataset_path -- root path
        traintest {[type]} -- [description]
        phase {[type]} -- [description]
        filetype {[type]} -- [description]

    Raises:
        Exception: [description]

    Returns:
        [type] -- [description]
    """
    folder = Path(f"{dataset_path}/underexpose_{traintest}")

    if filetype in ["click", "qtime"]:
        filename = f"underexpose_{traintest}_{filetype}-{phase}.{extension}"
    elif filetype in ["user", "item"]:
        filename = f"underexpose_{filetype}_feat.{extension}"
    elif filetype == "submit":
        filename = f"underexpose_{filetype}-{phase}.{extension}"
    else:
        raise Exception("filetype not recognized")
        
    return folder, filename

def get_raw_dataset(traintest="train", phase="0", filetype="click"):
    """gets dataset associated with arguments and does some cleaning

    Keyword Arguments:
        traintest {str} -- "train" or "test" (default: {"train"})
        phase {str} -- number between 0 and 6. (default: {"0"})
        filetype {str} -- one of "click", "item", "user", "qtime" (default: {"click"})
        
    if filetype is click, user or qtime, returns cleaned dataframe
    if filetype is item, returns the item_ids, text_vecs, and img_vecs
    """

    folder, filename= filename_from_args("dataset", traintest, phase, filetype, "csv")

    if filetype == "item":
        header = ["item_id","text_vec","img_vec"]
        with open(folder/filename) as handle:
            lines = handle.readlines()
            item_ids = np.zeros(len(lines), dtype=np.uint64)
            text_vec = np.zeros((len(lines), 128), dtype=np.float64)
            img_vec = np.zeros((len(lines), 128), dtype=np.float64)
            for i, line in tqdm(list(enumerate(lines))):
                line = literal_eval(line)
                item_ids[i] = line[0]
                text_vec[i] = line[1]
                img_vec[i] = line[2]
        return {"item_id": item_ids, "text_vec": text_vec, "img_vec": img_vec}

    if filetype == "click":
        header = ["user_id","item_id","time"]
        dataframe = pd.read_csv(folder/filename, sep=",", names=header)
        return dataframe
    elif filetype == "qtime":
        header = ["user_id", "time"]
        dataframe = pd.read_csv(folder/filename, sep=",", names=header)
        return dataframe
    elif filetype == "user":
        header = ["user_id","user_age_level","user_gender","user_city_level"]
        dataframe = pd.read_csv(folder/filename, sep=",", names=header)
        dataframe["user_gender"] = dataframe["user_gender"].map({"M": 1, "F":-1, None:0})

        dataset = {}
        dataset["user_id"] = np.array(dataframe["user_id"], dtype=np.uint64)
        dataset["user_age_level"] = np.array(dataframe["user_age_level"], dtype=np.float64)
        dataset["user_gender"] = np.array(dataframe["user_gender"], dtype=np.float64)
        dataset["user_city_level"] = np.array(dataframe["user_city_level"], dtype=np.float64)

        return dataset

traintest_vals = ["train", "test"]
phases = [str(i) for i in range(7)]

def fix_timestamps(dataset):
    """
        fixes timestamps for the dataframes such that 1 = 1 hour from the first timestamp
    """
    min_timestamp = 999
    length_of_day = 0.0000547 # got by staring at insights.ipynb
    length_of_hour = length_of_day/24

    print("normalizing timestamps")
    for traintest, phase, filetype in tqdm(product(traintest_vals, phases, ["click", "qtime"])):
        if traintest != "train" or filetype != "qtime":
            df = dataset[(traintest, phase, filetype)]
            min_timestamp = min(min_timestamp, min(df["time"]))

    # save timestamps to adjusted values
    for traintest, phase, filetype in tqdm(product(traintest_vals, phases, ["click", "qtime"])):
        if traintest != "train" or filetype != "qtime":
            df = dataset[(traintest, phase, filetype)]
            df["time"] = (df["time"]-min_timestamp)/length_of_hour

    return dataset

def build_contiguized_keymap(raw_ids, keymap=None, next_key=None):
    if keymap == None:
        keymap = {}
    if next_key == None:
        next_key = 0

    initial_keymap_size = len(keymap)
    new_items = set(raw_ids) - set(keymap.keys())
    for new_item in new_items:
        keymap[new_item] = next_key
        next_key += 1

    contiguized_ids = [keymap[raw_id] for raw_id in raw_ids]
    return contiguized_ids, (keymap, next_key)

def contiguize_dataset_keys(dataset):
    """
        contiguizing item and user keymaps such that each user_id and item_id could be used as an array index

        Arguments:
            dataset {dict of dataframes}
    """
    item_keymap, next_item_key = {}, 0
    user_keymap, next_user_key = {}, 0

    for phase in phases:
        for traintest in ["train", "test"]:
            df = dataset[(traintest, phase, "click")]
            df["item_id"], (item_keymap, next_item_key) = build_contiguized_keymap(df["item_id"], item_keymap, next_key=next_item_key)
            df["user_id"], (user_keymap, next_user_key) = build_contiguized_keymap(df["user_id"], user_keymap, next_key=next_user_key)

        df = dataset[("test", phase, "qtime")]
        df["user_id"], (user_keymap, next_user_key) = build_contiguized_keymap(df["user_id"], user_keymap, next_key=next_user_key)

    df = dataset["user"]
    df["user_id"], (user_keymap, next_user_key) = build_contiguized_keymap(df["user_id"], user_keymap, next_key=next_user_key)

    df = dataset["item"]
    df["item_id"], (item_keymap, next_item_key) = build_contiguized_keymap(df["item_id"], item_keymap, next_key=next_item_key)

    dataset["user_keymap"] = {v: k for k, v in user_keymap.items()}
    dataset["item_keymap"] = {v: k for k, v in item_keymap.items()}

    return dataset

def add_missing_users_items(dataset):
    users = dataset["user"]
    items = dataset["item"]

    uid_line_map = {uid: i for i, uid in enumerate(users["user_id"])}
    average_user = {key: np.mean(users[key][~pd.isnull(users[key])].astype(float)) for key in users.keys() if key != "user_id"}
    average_user["user_gender"] = 0
    for key in users.keys():
        if key != "user_id":
            users[key][pd.isnull(users[key])] = average_user[key]

    users_to_add = set(range(len(dataset["user_keymap"].keys()))) - set(uid_line_map.keys())
    num_users_to_add = len(users_to_add)
    for uid in users_to_add:
        assert (uid in dataset["user_keymap"])
    
    dataset["user"]["user_id"] = np.append(dataset["user"]["user_id"], list(users_to_add))
    age_levels_to_add = np.tile(average_user["user_age_level"], (num_users_to_add, 1))
    dataset["user"]["user_age_level"] = np.append(dataset["user"]["user_age_level"], age_levels_to_add)
    genders_to_add = np.tile(average_user["user_age_level"], (num_users_to_add, 1))
    dataset["user"]["user_gender"] = np.append(dataset["user"]["user_gender"], genders_to_add)
    city_levels_to_add = np.tile(average_user["user_age_level"], (num_users_to_add, 1))
    dataset["user"]["user_city_level"] = np.append(dataset["user"]["user_city_level"], city_levels_to_add)

    iid_line_map = {iid: i for i, iid in enumerate(items["item_id"])}
    average_item = {key: np.mean(items[key].astype(float), axis=0) for key in items.keys() if key != "item_id"}

    items_to_add = set(range(len(dataset["item_keymap"].keys()))) - set(iid_line_map.keys())
    num_items_to_add = len(items_to_add)
    for iid in items_to_add:
        assert (iid in dataset["item_keymap"])
    dataset["item"]["item_id"] = np.append(dataset["item"]["item_id"], list(items_to_add), axis=0)
    text_to_add = np.tile(average_item["text_vec"], (num_items_to_add, 1))
    dataset["item"]["text_vec"] = np.append(dataset["item"]["text_vec"], text_to_add, axis=0)
    imgs_to_add = np.tile(average_item["img_vec"], (num_items_to_add, 1))
    dataset["item"]["img_vec"] = np.append(dataset["item"]["img_vec"], imgs_to_add, axis=0)

    print(f"added {len(users_to_add)}/{len(dataset['user_keymap'])} to user dataset and {len(items_to_add)}/{len(dataset['item_keymap'])} items")
    return dataset

def build_validation_set(dataset):
    validation_df = dataset[("test", "0", "click")]
    for phase in tqdm([str(phase) for phase in range(1, 7)]):
        next_phase_df = dataset[("test", phase, "click")]
        validation_df = validation_df.append(next_phase_df)
        
    dataset["val_sessions"] = validation_df
    return dataset

def groupby_user(dataset, phases, name):
    print(f"grouping phases {phases} by user")
    user_full_sessions = defaultdict(lambda: [])
    for phase in tqdm(phases):
        df = dataset[phase]
        for _, row in tqdm(list(df.iterrows())):
            user_id = row["user_id"]
            user_full_sessions[user_id].append((row["time"], row["item_id"]))
        del dataset[phase]

    for uid in user_full_sessions.keys():
        user_full_sessions[uid].sort()

    dataset[name] = dict(user_full_sessions)
    return dataset

def split_full_sessions(dataset, phase, max_time_jump=24):
    print(f"splitting {phase} click dataset into sessions")
    ds_phase = dataset[phase]

    user_sessions = {}
    for uid in ds_phase.keys():
        user_sessions[uid] = []
        
        current_session_items = []
        current_session_times = []
        for time, item_id in ds_phase[uid]:
            if len(current_session_items) == 0:
                current_session_items.append(item_id)
                current_session_times.append(time)
            else:
                if current_session_items[-1] == item_id and current_session_times[-1] == time:
                    continue
                elif time - current_session_times[-1] < max_time_jump:
                    current_session_items.append(item_id)
                    current_session_times.append(time)
                else:
                    user_sessions[uid].append((current_session_items, current_session_times))
                    current_session_items = [item_id]
                    current_session_times = [time]
        user_sessions[uid].append((current_session_items, current_session_times))
    
    dataset[phase] = user_sessions
    return dataset

def train_session_to_X_y(user_id, session, augment=False):
    """splits each N length session into N-2 segments each with at least 2 items. drops sessions that start with only 2 items. returns the session split into shorter sessions all starting at 0. normalizes session times to be 1/whatever. last session item will be the y value and its time will be the original time.

    Arguments:
        user_id {int} -- the userid of the session
        session {[float], [float]} -- session_itemids, session_itemid_times

    """
    session_items = []
    session_item_times = []

    if len(session[0]) < 2:
        return [], session_items, session_item_times

    # i = 0
    for i in range(len(session[0])-1):
        for j in range(i+2, len(session[0])):
            session_items.append(np.array(session[0][i:j+1], dtype=np.int32))
            times = np.zeros(j-i+1, dtype=np.float64)
            times[:-1] = 1/(session[1][j]-session[1][i:j]+1)
            times[-1] = session[1][j]
            session_item_times.append(times)

    sessions_to_use = range(len(session_items)-1)
    sessions_to_use = random.sample(sessions_to_use, k=int(len(sessions_to_use)/2))
    sessions_to_use.append(len(sessions_to_use)-1)
    sessions_to_use = set(sessions_to_use)
    session_items = [time for i, time in enumerate(session_items) if i in sessions_to_use]
    session_item_times = [time for i, time in enumerate(session_item_times) if i in sessions_to_use]

    user_ids = np.tile(np.int32(user_id), len(session_items))

    return user_ids, session_items, session_item_times

def augment_w_short_sessions(dataset, phase):
    """
        augments dataset by splitting every session into every subsession of the session
    """
    print("augmenting with short sessions")
    user_ids = []
    sessions_items = []
    sessions_item_times = []

    for user_id, sessions in tqdm(list(dataset[phase].items())):
        for session in sessions:
            user_ids_p, session_items_p, session_item_times_p = train_session_to_X_y(user_id, session)
            user_ids.extend(user_ids_p)
            sessions_items.extend(session_items_p)
            sessions_item_times.extend(session_item_times_p)

    user_sessions = {}
    for user_id, session_items, session_item_times in zip(user_ids, sessions_items, sessions_item_times):
        if user_id not in user_sessions:
            user_sessions[user_id] = []
        user_sessions[user_id].append((session_items, session_item_times))

    dataset[phase] = user_sessions
    return dataset

def final_dataset_formatting(dataset):
    print("formatting the dataset to be ready for training")
    train_X, train_y = {"user_id": [], "session": [], "X_time": [], "y_time":[]}, []
    for uid, sessions in tqdm(random.shuffle(dataset["train_sessions"].items())):
        for session in sessions:
            train_X["user_id"].append(uid)
            train_X["session"].append(session[0][:-1])
            train_X["X_time"].append(session[1][:-1])
            train_X["y_time"].append(session[1][-1])
            train_y.append(session[0][-1])
    print("formatted training sessions")

    val_X, val_y = {"user_id": [], "session": [], "X_time": [], "y_time":[]}, []
    num_sessions, num_dropped_val_sessions = 0, 0
    for uid, sessions in tqdm(dataset["val_sessions"].items()):
        for session in sessions:
            num_sessions += 1
            if len(session) < 2:
                num_dropped_val_sessions += 1
                continue
            val_X["user_id"].append(uid)
            val_X["session"].append(session[0][:-1])
            val_X["X_time"].append(session[1][:-1])
            val_X["y_time"].append(session[1][-1])
            val_y.append(session[0][-1])
    if num_dropped_val_sessions > 0:
        print(f"dropped {num_dropped_val_sessions}/{num_sessions} of all validation sessions")
    print("formatted val sessions")

    test_X = {}
    for phase in phases:
        test_X[phase] = {"user_id": [], "session": [], "X_time": [], "y_time":[]}
        click_df = dataset[("test", phase, "click")]
        qtime_df = dataset[("test", phase, "qtime")]
        y_user_time = {uid: time for uid, time in zip(qtime_df["user_id"], qtime_df["time"])}
        
        for uid, session in click_df.groupby("user_id"):
            session = session.sort_values(by="time")
            y_time = y_user_time[uid]
            test_X[phase]["user_id"].append(uid)
            test_X[phase]["session"].append(session["item_id"])
            
            test_X[phase]["X_time"].append(1/(y_time-session["time"]+1))
            test_X[phase]["y_time"].append(y_time)
    print("formatted test sessions")


    train = (train_X, train_y)
    val = (val_X, val_y)
    test = (test_X,)
    user_item_info = (dataset["user"], dataset["item"])
    keymaps = dataset["user_keymap"], dataset["item_keymap"]

    return (train, val, test, user_item_info, keymaps)

if __name__ == "__main__":
    dataset = {}
    for traintest, phase, filetype in tqdm(product(traintest_vals, phases, ["click", "qtime"])):
        if traintest != "train" or filetype != "qtime":
            df = get_raw_dataset(traintest=traintest, phase=phase, filetype=filetype)
            dataset[(traintest, phase, filetype)] = df
    dataset["item"] = get_raw_dataset(traintest="train", filetype="item")
    dataset["user"] = get_raw_dataset(traintest="train", filetype="user")

    dataset = fix_timestamps(dataset)

    dataset = contiguize_dataset_keys(dataset)

    dataset = add_missing_users_items(dataset)

    if validation_set_source == "test":

        dataset = build_validation_set(dataset)

        train_phases = [("train", str(phase), "click") for phase in range(7)]
        dataset = groupby_user(dataset, train_phases, "train_sessions")
        dataset = groupby_user(dataset, ["val_sessions"], "val_sessions")

    if validation_set_source[0] == "phase":
        phases = set(range(7))
        phases -= {validation_set_source[1]}
        train_phases = [("train", str(phase), "click") for phase in phases]
        dataset = groupby_user(dataset, train_phases, "train_sessions")
        train_phases = [("train", validation_set_source[1], "click")]
        dataset = groupby_user(dataset, train_phases, "val_sessions")

    print("pre split full sessions")
    # print(dataset["train_sessions"][0])

    dataset = split_full_sessions(dataset, "train_sessions")
    print("post split full sessions")
    # print(dataset["train_sessions"][0])

    # with open(f"temp_dataset.pkl", "wb+") as handle:
    #     pickle.dump(dataset, handle)

    # with open(f"temp_dataset.pkl", "rb") as handle:
    #     dataset = pickle.load(handle)

    dataset = augment_w_short_sessions(dataset, "train_sessions")
    print("post augment w short sessions")
    # print(dataset["train_sessions"][0])

    dataset["val_sessions"] = {uid: [session] for uid, session in dataset["val_sessions"].items()}

    dataset = final_dataset_formatting(dataset)
    print("formatted dataset")

    os.makedirs(Path("processed_data"), exist_ok=True)
    print(f'saving at {Path("processed_data")/f"dataset.pkl"}')
    with bz2.open(Path("processed_data")/f"dataset.pkl", "wb") as handle:
        pickle.dump(dataset, handle)


# notes on data processing
    # 1. cleans nulls with average values. replaces strings with appropriate values.
    # 2. fixes timestamps
    # 3. normalzie user and item information