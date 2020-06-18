import dill as pickle
from pathlib import Path
import csv
from tqdm import tqdm
import numpy as np
from collections import defaultdict

# internal movieids are used as movieids aren't contiguous

tag_names = Path("dataset")/"genome-tags.csv"                   # tag name lookup
movie_review_relevance = Path("dataset")/"genome-scores.csv"    # movieid/tagid/relevance
movie_genres = Path("dataset")/"movies.csv"                     # movieid/movie title/genres
reviews = Path("dataset")/"tags_shuffled_rehashed.csv"          # userid/movieid/tag
train_set = Path("dataset")/"train_ratings_binary.csv"          # train set - userid/movieid/ratings
val_set = Path("dataset")/"val_ratings_binary.csv"              # val set - userid/movieid/ratings
test_set = Path("dataset")/"test_ratings.csv"                   # test set - userid/movieids

NUM_MOVIES = 27278
NUM_USERS = 138493
NUM_TRAINING_SET = 11946576
NUM_TAGS = 1128
NUM_GENRES = 20

ALL_GENRES = ['Drama', 'Comedy', 'Thriller', 'Romance', 'Action', 'Crime', 'Horror', 'Documentary', 'Adventure', 'Sci-Fi', 'Mystery', 'Fantasy', 'War', 'Children', 'Musical', 'Animation', 'Western', 'Film-Noir', 'none/other', 'IMAX']

def get_movieid_mid_lookup(recompute=False):
    if recompute:
        movieid_mid_lookup = {}
        def add_movieids_to_lookuptable(filename):
            print(f"updating lookuptable with mids from {filename}")
            with open(filename, newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                for rating in tqdm(reader):
                    movieid = int(float(rating["movieId"]))
                    if movieid not in movieid_mid_lookup:
                        movieid_mid_lookup[movieid] = add_movieids_to_lookuptable.next_unassigned_mid
                        add_movieids_to_lookuptable.next_unassigned_mid += 1
        add_movieids_to_lookuptable.next_unassigned_mid = 0

        add_movieids_to_lookuptable(train_set)
        add_movieids_to_lookuptable(val_set)
        add_movieids_to_lookuptable(test_set)
        add_movieids_to_lookuptable(movie_genres)
        add_movieids_to_lookuptable(movie_review_relevance)

        with open("movieid_mid_lookup.pickle", "wb+") as lookup_file:
            pickle.dump(movieid_mid_lookup, lookup_file)

        return movieid_mid_lookup

    else:
        with open("movieid_mid_lookup.pickle", "rb") as lookup_file:
            movieid_mid_lookup = pickle.load(lookup_file)
        return movieid_mid_lookup

userid_uid_lookup = lambda userid: userid-1

def genre_parser(genre):
    if genre == "(no genres listed)":
        return ["none/other"]
    return genre.split("|")

def get_dataset(filename, include_ys=True, recompute=False):
    # depends on get_movieid_mid_lookup
    print(f"retrieving dataset from {filename}")
    movieid_mid_lookup = get_movieid_mid_lookup()

    cache_filename = str(filename)[:-len(filename.suffix)]+".pickle"
    if recompute:
        with open(filename, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            user_Xs, movie_Xs, ys = [], [], []
            for rating in tqdm(reader):
                userid = int(float(rating["userId"]))
                uid = userid_uid_lookup(userid)
                user_Xs.append(uid)
                
                movieid = int(float(rating["movieId"]))
                mid = movieid_mid_lookup[movieid]
                movie_Xs.append(mid)

                if include_ys:
                    score = [1, 0] if (rating["rating"] == "1") else [0, 1]
                    ys.append(score)
            user_Xs = np.array(user_Xs).reshape(-1, 1)
            movie_Xs = np.array(movie_Xs).reshape(-1, 1)
            if include_ys:
                ys = np.array(ys).reshape(-1, 2)
        with open(cache_filename, "wb+") as cache_file:
            if include_ys:
                pickle.dump((user_Xs, movie_Xs, ys), cache_file)
            else:
                pickle.dump((user_Xs, movie_Xs), cache_file)
    else:
        with open(cache_filename, "rb") as cache_file:
            if include_ys:
                user_Xs, movie_Xs, ys = pickle.load(cache_file)
            else:
                user_Xs, movie_Xs = pickle.load(cache_file)
    
    if include_ys:
        return user_Xs, movie_Xs, ys
    else:
        return user_Xs, movie_Xs

def get_movie_genres_one_hot(recompute=False):
    # depends on get_movieid_mid_lookup
    if recompute:
        movieid_mid_lookup = get_movieid_mid_lookup()
        with open(movie_genres, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            movie_genres_one_hot = {movieid_mid_lookup[int(float(movie["movieId"]))]: np.array([genre in movie["genres"] for genre in ALL_GENRES]) for movie in reader}

        with open("mid_genres_one_hot.pickle", "wb+") as genre_file:
            pickle.dump(movie_genres_one_hot, genre_file)
    else:
        with open("mid_genres_one_hot.pickle", "rb") as genre_file:
            movie_genres_one_hot = pickle.load(genre_file)

    return movie_genres_one_hot


def get_dataset_genres(dataset_filename, dataset_includes_ys, recompute=False):
    # depends on get_dataset, get_movie_genres_one_hot
    cache_filename = str(dataset_filename)[:-len(dataset_filename.suffix)]+"_genres.pickle"
    if recompute:
        dataset = get_dataset(dataset_filename, include_ys=dataset_includes_ys, recompute=False)
        movie_Xs = dataset[1]

        movie_genres_one_hot = get_movie_genres_one_hot(recompute=True)

        genres = np.array([movie_genres_one_hot[x[0]] for x in tqdm(movie_Xs)])
        with open(cache_filename, "wb+") as genre_file:
            pickle.dump(genres, genre_file)
    else:
        with open(cache_filename, "rb") as genre_file:
            genres = pickle.load(genre_file)
    return genres

def batchify(*args, batch_size=1000, shuffle=True, arg_len=None):
    if batch_size == -1:
        yield args

    num_elems = len(args[0]) if arg_len == None else arg_len

    if shuffle:
        for arg in args:
            assert(type(arg) == np.ndarray)
        shuffle_indices = np.arange(num_elems, dtype=np.int64).astype(int)
        np.random.shuffle(shuffle_indices)
        for i in range(0, num_elems, batch_size):
            array_indices = shuffle_indices[i: i+batch_size]
            try:
                yield [arg[array_indices] for arg in args]
            except:
                raise Exception("args to batchify must be numpy arrays if shuffle True")
    else:
        for i in range(0, num_elems, batch_size):
            yield [arg[i: i+batch_size] for arg in args]

tagid_to_tid = lambda x: x-1


def get_raw_tag_relevances():
    print('loading movie tags from csv')
    movieid_mid_lookup = get_movieid_mid_lookup()
    tags = np.zeros((NUM_MOVIES, NUM_TAGS), dtype=np.float16)

    with open(movie_review_relevance, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for rating in tqdm(reader):
            movieid = int(float(rating["movieId"]))
            tagid = int(float(rating["tagId"]))
            relevance = float(rating["relevance"])

            mid = movieid_mid_lookup[movieid]
            tid = tagid_to_tid(tagid)
            tags[mid, tid] = relevance

    nonzero_mids = []
    nonzero_relevances = []

    for i, row in enumerate(tags):
        if 0 not in row:
            nonzero_mids.append(i)
            nonzero_relevances.append(row)

    nonzero_mids = np.array(nonzero_mids)
    nonzero_relevances = np.array(nonzero_relevances)

    return nonzero_mids, nonzero_relevances

def build_pca_model(preserved_variance, recompute):
    if not recompute:
        try:
            with open("mid_to_proj_tag.pickle", "rb") as mid_proj_tag_file:
                mid_to_proj_tag = pickle.load(mid_proj_tag_file)
        except:
            recompute = True

    if recompute:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        # scale data
        movie_tag_mids, movie_tag_relevances = get_raw_tag_relevances()
        print(f"rebuilding pca model preserving {preserved_variance} variance")

        if preserved_variance < 1: # only use pca if not preserving all variance
            scaler = StandardScaler()
            scaler.fit(movie_tag_relevances)
            movie_tag_relevances = scaler.transform(movie_tag_relevances)

            # build model & transform data
            model = PCA(preserved_variance).fit(movie_tag_relevances)
            movie_tag_relevances = model.transform(movie_tag_relevances)

        # normalizing data to [0, 1]
        print("normalizing")
        flat_tag_rel = movie_tag_relevances.flatten()
        min_val, max_val = min(flat_tag_rel), max(flat_tag_rel)
        movie_tag_relevances = (movie_tag_relevances-min_val)/(max_val-min_val)

        print("setting default value to average")
        avg = np.mean(movie_tag_relevances, axis=0)

        proj_tag_dim = movie_tag_relevances[0].shape[0]
        mid_to_proj_tag = defaultdict(lambda: np.full((proj_tag_dim,), avg))
        for mid, tag in zip(movie_tag_mids, movie_tag_relevances):
            mid_to_proj_tag[mid] = tag

        with open("mid_to_proj_tag.pickle", "wb+") as mid_proj_tag_file:
            pickle.dump(mid_to_proj_tag, mid_proj_tag_file)

    return mid_to_proj_tag

def get_tags(mids, preserved_variance, recompute=False):
    mid_to_proj_tag = build_pca_model(preserved_variance, recompute)
    out = np.zeros((len(mids), len(mid_to_proj_tag[mids[0][0]])), dtype=np.float16)
    print("getting tags")
    for i, mid in enumerate(mids):
        out[i] = mid_to_proj_tag[mid[0]]
    return out


def get_movies_with_missing_tags():
    movie_mids_no_0, _ = get_movie_tag_relevances()
    movieid_mid_lookup = get_movieid_mid_lookup()
    movie_mids_no_0 = set(movie_mids_no_0)
    
    movies_with_missing_tags = []
    for _, mid in movieid_mid_lookup.items():
        if mid not in movie_mids_no_0:
            movies_with_missing_tags.append(mid)

    return movies_with_missing_tags

def get_tag_to_tid(recompute=False):
    if not recompute:
        try:
            with open("tag_to_tid.pickle", "rb") as tag_to_tid_file:
                tag_to_tid = pickle.load(tag_to_tid_file)
        except:
            recompute = True
    if recompute:
        tag_to_tid = {}
        with open(tag_names, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                tagid = int(float(row["tagId"]))
                tag = row["tag"]
                tid = tagid_to_tid(tagid)
                tag_to_tid[tag] = tid
        with open("tag_to_tid.pickle", "wb+") as tag_to_tid_file:
                pickle.dump(tag_to_tid, tag_to_tid_file)
    return tag_to_tid