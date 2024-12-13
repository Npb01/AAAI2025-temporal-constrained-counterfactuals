def calculate_fitness(org_prefix_flat, flattened_prefixes, onehot_cols_start_ind, max_trace_len_all,
                      features_list, seq_cols_start_ind, max_ED, target_class, blackbox, alpha=0.8, skip_ED=True):
    jaccard_dist = DistanceMetric.get_metric('jaccard')
    euclidean_dist = DistanceMetric.get_metric('euclidean')

    num_cols = [f_p[: onehot_cols_start_ind] for f_p in flattened_prefixes]
    num_cols.insert(0, org_prefix_flat[: onehot_cols_start_ind])
    num_dist = euclidean_dist.pairwise(num_cols)[0, 1:]

    onehot_cols = [f_p[onehot_cols_start_ind:seq_cols_start_ind] for f_p in flattened_prefixes]
    onehot_cols.insert(0, org_prefix_flat[onehot_cols_start_ind:seq_cols_start_ind])
    onehot_dist = jaccard_dist.pairwise(onehot_cols)[0, 1:]

    restored_prefixes = np.array([transform_prefix(f_p, seq_cols_start_ind, features_list,
                                                   max_trace_len_all, restore=True)
                                  for f_p in flattened_prefixes])
    restored_prefixes_class = np.argmax(blackbox.predict(restored_prefixes), axis=-1)

    indicator_1 = restored_prefixes_class == target_class

    indicator_2 = np.array([np.array_equal(org_prefix_flat[:seq_cols_start_ind], f_p[:seq_cols_start_ind])
                            for f_p in flattened_prefixes])

    if skip_ED:
        fitness = indicator_1 + (1 - ((num_dist + onehot_dist) / 2)) - indicator_2
    else:
        ED = np.array([get_ED(org_prefix_flat[seq_cols_start_ind:], f_p[seq_cols_start_ind:])
                       for f_p in flattened_prefixes]) * (alpha / max_ED)
        fitness = indicator_1 + (1 - ((num_dist + onehot_dist + ED) / 3)) - indicator_2

    return fitness

def crossover_mutate(next_pop, features_map, features_map_mut, kde, uniq_seq, pc=0.7, pm= 0.2, shuffle=True):
    """
        next_pop: the list/array of next population after selection
        features_map: the indexes of each feature
            bpic2017 features_map: [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], 
            [6, 20], [20, 22], [22, 23], [23, 24], [24, 25], [25, 26], [26, None]]
            bpic2017 features_map_mut: [[0, 6], [6, 22], [22, 26], [26, None]]
                        assuming numerical columns first, followed by 1hot encoded columns and seq columns.

        return: the untouched next_pop and the crossovered instances, the size is set to be 2*len(next_pop)
    """
    parents_size = ceil(len(next_pop)*pc)
    parents_ind = np.random.choice(range(len(next_pop)), size=parents_size, replace=False)
    parents = np.array(next_pop)[np.random.choice(parents_ind, size=(parents_size*2, 2), replace=True)]    

    # uniform crossover
    crossovered = [np.concatenate([p[v][features_map[i][0]:features_map[i][1]] 
                                   for i, v in enumerate(np.random.randint(0, 2, size=len(features_map)))]) 
                   for p in parents]
    crossovered = np.array(crossovered, dtype=object)

    # mutation
    mutation_size = ceil(crossovered.shape[0]*pm)

    mutation_ind = np.random.choice(range(crossovered.shape[0]), size=mutation_size, replace=False)
    non_mutation_ind = [i for i in range(crossovered.shape[0]) if i not in mutation_ind]
    mutated_children = crossovered.copy()[mutation_ind]

    mutated_non_seq_attri = np.array([c[:features_map_mut[-1][0]] for c in mutated_children])

    mutated_1hot_attri = [np.eye((features_map_mut[i][1] - 
                                 features_map_mut[i][0]))[np.random.choice((features_map_mut[i][1] - 
                                                                          features_map_mut[i][0]), 
                                                                          mutation_size)] 
                          for i in range(1, len(features_map_mut) -1 )]

    mutated_1hot_attri = np.concatenate(mutated_1hot_attri, axis=1)

    non_seq_attri = np.concatenate([kde.sample(mutation_size), mutated_1hot_attri], axis=1)

    mutated_seq_attri = np.array(uniq_seq, dtype=object)[np.random.randint(len(uniq_seq), size=mutation_size)]

    mutated_non_seq_attri[non_seq_attri >= 0] = non_seq_attri[non_seq_attri >= 0]
    mutated_children = np.array([np.concatenate([v, mutated_seq_attri[i]]) 
                                 for i, v in enumerate(mutated_non_seq_attri)], dtype=object)

    non_mutated_children = np.array([c for i, c in enumerate(crossovered) if i in non_mutation_ind], dtype=object)

    # crossovered[mutation_ind] = mutated_children

    new_next_pop = np.array(list(next_pop) + list(mutated_children) + list(non_mutated_children), dtype=object)

    if shuffle:
        np.random.shuffle(new_next_pop)

    return new_next_pop


def init_first_pop(prefix_of_interest, traces_l, max_ED, max_trace_len_all, target_population,
                   y_start_index, seq_cols_start_ind, blackbox, features_list, return_org=True):
    prefixes, uniq_seq = get_first_pop(trace=prefix_of_interest,
                                       traces_l=traces_l,
                                       max_ED=max_ED,
                                       max_trace_len_all=max_trace_len_all,
                                       y_start_index=y_start_index,
                                       features_list=features_list,
                                       seq_cols_start_ind=seq_cols_start_ind,
                                       blackbox=blackbox,
                                       target_label=None)

    init_pop = np.tile(prefixes, (ceil(target_population * 2 / prefixes.shape[0]), 1, 1))
    flattened_first_pop = [transform_prefix(p,
                                            seq_cols_start_ind=seq_cols_start_ind,
                                            features_list=features_list,
                                            max_trace_len_all=max_trace_len_all,
                                            restore=False)
                           for p in init_pop]

    if return_org:
        y_first_pop_no_dup = np.argmax(blackbox.predict(prefixes), axis=-1)
        flattened_first_pop_no_dup = [transform_prefix(p,
                                                       seq_cols_start_ind=seq_cols_start_ind,
                                                       features_list=features_list,
                                                       max_trace_len_all=max_trace_len_all,
                                                       restore=False)
                                      for p in prefixes]
        return flattened_first_pop, uniq_seq, flattened_first_pop_no_dup, y_first_pop_no_dup
    else:
        return flattened_first_pop, uniq_seq


def run_GA_loop(n_generations, target_population, pc, pm, org_prefix_flat, flattened_prefixes, features_list,
                onehot_cols_start_ind, seq_cols_start_ind, max_ED, target_class, blackbox,
                features_map, features_map_mut, uniq_seq, kde, max_trace_len_all, skip_ED=True):
    middle_pop_size = round(target_population * 1.5)

    for i in range(n_generations):
        # print(i, 'generation')
        # Selection
        fitness = calculate_fitness(org_prefix_flat=org_prefix_flat,
                                    flattened_prefixes=flattened_prefixes,
                                    onehot_cols_start_ind=onehot_cols_start_ind,
                                    max_trace_len_all=max_trace_len_all,
                                    features_list=features_list,
                                    seq_cols_start_ind=seq_cols_start_ind,
                                    max_ED=max_ED,
                                    target_class=target_class,
                                    blackbox=blackbox,
                                    skip_ED=skip_ED,
                                    )

        fitness_cutoff = np.sort(fitness)[-middle_pop_size]
        next_pop_ind = fitness >= fitness_cutoff

        next_pop = np.array([next_c for i, next_c in enumerate(flattened_prefixes) if next_pop_ind[i]], dtype=object)
        next_pop = np.array([np.array(s) for s in set(tuple(s) for s in next_pop)], dtype=object)
        if len(next_pop) > middle_pop_size:
            next_pop = next_pop[: middle_pop_size]
        # print("unique next pop:", len([np.array(s) for s in set(tuple(s) for s in next_pop)]))
        # print("unique_next_pop_no_seq:", np.unique(np.array([n_p[:26] for n_p in next_pop]), axis=0).shape[0])
        # print("unique_fitness:", np.unique(fitness).shape[0])
        # print("avg_fitness:", np.mean(fitness))

        if len(next_pop) < target_population:
            next_pop = np.concatenate([next_pop,
                                       next_pop[np.random.randint(len(next_pop),
                                                                  size=target_population - len(next_pop))]])
        # Crossover&Mutation
        next_pop_before_selection = crossover_mutate(next_pop=next_pop,
                                                     features_map=features_map,
                                                     features_map_mut=features_map_mut,
                                                     kde=kde,
                                                     uniq_seq=uniq_seq,
                                                     pc=pc,
                                                     pm=pm,
                                                     shuffle=True)

        flattened_prefixes = next_pop_before_selection
        # print("next_pop:", next_pop.shape[0])
        # print("next_pop_before_selection: ", next_pop_before_selection.shape[0])

        if (((i + 1) == n_generations) or np.mean(fitness) >= 1.5):
            # if (((i + 1) == n_generations)):
            flattened_prefixes = np.array([np.array(s) for s in set(tuple(s) for s in flattened_prefixes)],
                                          dtype=object)
            fitness = calculate_fitness(org_prefix_flat=org_prefix_flat,
                                        flattened_prefixes=flattened_prefixes,
                                        onehot_cols_start_ind=onehot_cols_start_ind,
                                        max_trace_len_all=max_trace_len_all,
                                        features_list=features_list,
                                        seq_cols_start_ind=seq_cols_start_ind,
                                        max_ED=max_ED,
                                        target_class=target_class,
                                        blackbox=blackbox,
                                        skip_ED=skip_ED)

            if len(fitness) > target_population:
                fitness_cutoff = np.sort(fitness)[-round(target_population)]
                final_pop_ind = fitness >= fitness_cutoff
                final_pop = np.array([next_c for i, next_c in enumerate(flattened_prefixes) if final_pop_ind[i]],
                                     dtype=object)
                if len(final_pop) > target_population:
                    final_pop = final_pop[:target_population]
                # print("---unique final pop:", len([np.array(s) for s in set(tuple(s) for s in final_pop)]))
                return final_pop
            else:
                return flattened_prefixes
def do_random_init(encoder, num_inits, features_to_vary, query_instance, desired_class, desired_range,rng,categorical_features,continous_features, cat_feature_index, cont_feature_index, feature_names):
    valid_inits = []
    #precisions = self.data_interface.get_decimal_precisions()
    #rng.bit_generator.state = np.random.PCG64(self.random_seed).state
    while len(valid_inits) < num_inits:
        num_remaining = num_inits - len(valid_inits)
        num_features = query_instance.shape[1]
        print(num_features)
        # Generate random initializations for all features at once
        random_inits = np.zeros((num_remaining, num_features))
        for jx, feature in enumerate(feature_names):
            print(jx,feature)
            if feature in features_to_vary:
                if feature in continous_features:
                    random_inits[:, jx] = rng.uniform(0, 1, num_remaining)
                    random_inits[:, jx] = np.round(random_inits[:, jx])
                else:
                    random_inits[:, jx] = rng.choice(list(encoder._label_dict[feature].values()), num_remaining)
            else:
                random_inits[:, jx] = query_instance[:,jx]
        if self.model.model_type == ModelTypes.Classifier:
            valid_mask = np.apply_along_axis(is_cf_valid, 1, self.predict_fn_scores(random_inits))
        else:
            valid_mask = np.apply_along_axis(is_cf_valid, 1, self.predict_fn_scores(random_inits)[:,np.newaxis])
        valid_inits.extend(random_inits)
    return np.array(valid_inits[:num_inits])
def is_cf_valid(self, model_score):
    """Check if a cf belongs to the target class or target range.
    """
    # Converting to single prediction if the prediction is provided as a
    # singleton array
    correct_dim = 1 if self.model.model_type == ModelTypes.Classifier else 0
    if hasattr(model_score, "shape") and len(model_score.shape) > correct_dim:
        model_score = model_score[0]
    # Converting target_cf_class to a scalar (tf/torch have it as (1,1) shape)
    if self.model.model_type == ModelTypes.Classifier:
        target_cf_class = self.target_cf_class
        if hasattr(self.target_cf_class, "shape"):
            if len(self.target_cf_class.shape) == 1:
                target_cf_class = self.target_cf_class[0]
            elif len(self.target_cf_class.shape) == 2:
                target_cf_class = self.target_cf_class[0][0]
        target_cf_class = int(target_cf_class)

        if len(model_score) == 1:  # for tensorflow/pytorch models
            pred_1 = model_score[0]
            validity = True if \
                ((target_cf_class == 0 and pred_1 <= self.stopping_threshold) or
                 (target_cf_class == 1 and pred_1 >= self.stopping_threshold)) else False
            return validity
        elif len(model_score) == 2:  # binary
            pred_1 = model_score[1]
            validity = True if \
                ((target_cf_class == 0 and pred_1 <= self.stopping_threshold) or
                 (target_cf_class == 1 and pred_1 >= self.stopping_threshold)) else False
            return validity
        else:  # multiclass
            return np.argmax(model_score) == target_cf_class
    else:
        return self.target_cf_range[0] <= model_score and model_score <= self.target_cf_range[1]
def get_output(predictive_model, input_instance, model_score=True):
    """returns prediction probabilities for a classifier and the predicted output for a regressor.

    :returns: an array of output scores for a classifier, and a singleton
    array of predicted value for a regressor.
    """
    if model_score:
        return predictive_model.model.predict_proba(input_instance)
    else:
        try:
            return predictive_model.model.predict(input_instance)
        except:
            return predictive_model.model.predict(input_instance.astype('category'))

def is_cf_valid(model, model_score, target_cf_class, stopping_threshold, target_cf_range):
    """Check if a cf belongs to the target class or target range."""
    # Converting to single prediction if the prediction is provided as a
    # singleton array
    correct_dim = 1
    if hasattr(model_score, "shape") and len(model_score.shape) > correct_dim:
        model_score = model_score[0]
    # Converting target_cf_class to a scalar (tf/torch have it as (1,1) shape)
    if model.model_type == ModelTypes.Classifier:
        if hasattr(target_cf_class, "shape"):
            if len(target_cf_class.shape) == 1:
                target_cf_class = target_cf_class[0]
            elif len(target_cf_class.shape) == 2:
                target_cf_class = target_cf_class[0][0]
        target_cf_class = int(target_cf_class)

        if len(model_score) == 1:  # for tensorflow/pytorch models
            pred_1 = model_score[0]
            validity = True if \
                ((target_cf_class == 0 and pred_1 <= stopping_threshold) or
                 (target_cf_class == 1 and pred_1 >= stopping_threshold)) else False
            return validity
        elif len(model_score) == 2:  # binary
            pred_1 = model_score[1]
            validity = True if \
                ((target_cf_class == 0 and pred_1 <= stopping_threshold) or
                 (target_cf_class == 1 and pred_1 >= stopping_threshold)) else False
            return validity
        else:  # multiclass
            return np.argmax(model_score) == target_cf_class
    else:
        return target_cf_range[0] <= model_score and model_score <= target_cf_range[1]
