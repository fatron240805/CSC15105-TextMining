def group_suspicious_matches(matches, max_suspicious_gap: int = 2):
    '''
    Group matches by suspicious sentence position.

    Suspicious sentences belong to the same group when
    the distance between consecutive suspicious indices
    is <= max_suspicious_gap.

    Example:
        S0, S8, S17, S18, S36, S60, S61, S62

    becomes:
        [S0]
        [S8]
        [S17, S18]
        [S36]
        [S60, S61, S62]
    '''

    if not matches:
        return []

    matches_by_suspicious = {}

    for match in matches:
        suspicious_idx = match["suspicious_idx"]

        if suspicious_idx not in matches_by_suspicious:
            matches_by_suspicious[suspicious_idx] = []

        matches_by_suspicious[suspicious_idx].append(match)

    suspicious_indices = sorted(matches_by_suspicious.keys())
    groups = []
    current_group = [suspicious_indices[0]]

    for suspicious_idx in suspicious_indices[1:]:
        previous_idx = current_group[-1]

        if (suspicious_idx - previous_idx <= max_suspicious_gap):
            current_group.append(suspicious_idx)
        else:
            groups.append({
                "suspicious_indices": current_group,
                "matches": [
                    match
                    for idx in current_group
                    for match in matches_by_suspicious[idx]
                ],
            })
            current_group = [suspicious_idx]

    groups.append({
        "suspicious_indices": current_group,
        "matches": [
            match
            for idx in current_group
            for match in matches_by_suspicious[idx]
        ],
    })

    return groups


def merge_source_matches(matches, max_source_gap: int = 5):
    '''
    Merge source matches that are close to each other
    for the same suspicious sentence.

    Example:
        S8 -> T206
        S8 -> T207
        S8 -> T254
    becomes:
        S8 -> [T206, T207]
        S8 -> [T254]
    '''

    if not matches:
        return []

    matches_by_suspicious = {}

    for match in matches:
        suspicious_idx = (match["suspicious_idx"])

        if suspicious_idx not in matches_by_suspicious:
            matches_by_suspicious[suspicious_idx] = []

        matches_by_suspicious[suspicious_idx].append(match)

    merged = []

    for suspicious_idx in sorted(matches_by_suspicious.keys()):
        suspicious_matches = sorted(
            matches_by_suspicious[
                suspicious_idx
            ],
            key=lambda match: match[
                "source_idx"
            ],
        )

        current = [suspicious_matches[0]]

        for match in suspicious_matches[1:]:
            previous = current[-1]
            source_gap = (match["source_idx"] - previous["source_idx"])

            if source_gap <= max_source_gap:
                current.append(match)

            else:
                merged.append({
                    "suspicious_idx": suspicious_idx,
                    "source_start": current[0][
                        "source_idx"
                    ],
                    "source_end": current[-1][
                        "source_idx"
                    ],
                    "matches": current,
                    "score": sum(
                        item["score"]
                        for item in current
                    ) / len(current),
                })

                current = [match]

        merged.append({
            "suspicious_idx": suspicious_idx,
            "source_start": current[0][
                "source_idx"
            ],
            "source_end": current[-1][
                "source_idx"
            ],
            "matches": current,
            "score": sum(
                item["score"]
                for item in current
            ) / len(current),
        })

    return merged


def build_source_paths(
    suspicious_group,
    max_source_gap: int = 5,
):
    '''
    Build source paths inside one suspicious group.

    A path connects source groups from consecutive
    suspicious sentences when their source positions
    are increasing and close enough.

    Example:
        S60 -> 290
        S61 -> 291
        S62 -> 292

    creates:
        S60 -> T290
        S61 -> T291
        S62 -> T292
    '''

    suspicious_indices = (suspicious_group["suspicious_indices"])
    matches = suspicious_group["matches"]

    source_groups = merge_source_matches(
        matches,
        max_source_gap=max_source_gap,
    )

    groups_by_suspicious = {}
    for source_group in source_groups:
        suspicious_idx = (source_group["suspicious_idx"])

        if (
            suspicious_idx
            not in groups_by_suspicious
        ):
            groups_by_suspicious[
                suspicious_idx
            ] = []

        groups_by_suspicious[
            suspicious_idx
        ].append(source_group)

    paths = []
    first_suspicious = suspicious_indices[0]

    for source_group in groups_by_suspicious.get(first_suspicious, []):
        paths.append({
            "suspicious_indices": [
                first_suspicious
            ],
            "source_groups": [
                source_group
            ],
            "score": source_group["score"],
        })

    for suspicious_idx in suspicious_indices[1:]:

        candidates = groups_by_suspicious.get(
            suspicious_idx,
            [],
        )

        new_paths = []

        for path in paths:

            previous_group = path[
                "source_groups"
            ][-1]

            previous_source_end = (
                previous_group["source_end"]
            )

            extended = False

            for source_group in candidates:

                current_source_start = (
                    source_group["source_start"]
                )

                source_gap = (
                    current_source_start
                    - previous_source_end
                )

                # Source position must move forward.
                if current_source_start <= (
                    previous_source_end
                ):
                    continue

                # Source groups must be close.
                if source_gap > max_source_gap:
                    continue

                new_path = {
                    "suspicious_indices": (
                        path[
                            "suspicious_indices"
                        ]
                        + [suspicious_idx]
                    ),
                    "source_groups": (
                        path[
                            "source_groups"
                        ]
                        + [source_group]
                    ),
                }

                scores = [
                    group["score"]
                    for group in new_path[
                        "source_groups"
                    ]
                ]

                new_path["score"] = (
                    sum(scores) / len(scores)
                )

                new_paths.append(
                    new_path
                )

                extended = True

            # If this path cannot continue,
            # keep the old path as it is.
            if not extended:
                new_paths.append(path)

        paths = new_paths

    return paths


def select_best_paths(paths, min_path_length: int = 2):
    '''
    Select the best source path(s).

    Longer paths are preferred first.
    Among paths with the same length,
    higher average similarity is preferred.

    Only the best path is returned for now.
    '''

    if not paths:
        return []

    valid_paths = [
        path
        for path in paths
        if len(
            path["suspicious_indices"]
        ) >= min_path_length
    ]

    if not valid_paths:
        return []

    valid_paths = sorted(
        valid_paths,
        key=lambda path: (
            len(
                path[
                    "suspicious_indices"
                ]
            ),
            path["score"],
        ),
        reverse=True,
    )

    best_path = valid_paths[0]

    return [best_path]


def align_matches(
    matches,
    max_suspicious_gap: int = 2,
    max_source_gap: int = 5,
    min_path_length: int = 2,
):
    '''
    Perform source-path based alignment.

    Pipeline:
        candidate matches
            ->
        suspicious grouping
            ->
        source grouping
            ->
        source path construction
            ->
        best path selection

    Returns:
        List of alignment paths.
    '''

    if not matches:
        return []

    suspicious_groups = group_suspicious_matches(
        matches,
        max_suspicious_gap=max_suspicious_gap,
    )

    alignments = []

    for suspicious_group in suspicious_groups:

        paths = build_source_paths(
            suspicious_group,
            max_source_gap=max_source_gap,
        )

        best_paths = select_best_paths(
            paths,
            min_path_length=min_path_length,
        )

        if len(
            suspicious_group[
                "suspicious_indices"
            ]
        ) == 1:

            source_groups = (
                merge_source_matches(
                    suspicious_group["matches"],
                    max_source_gap=max_source_gap,
                )
            )

            if source_groups:

                best_source_group = max(
                    source_groups,
                    key=lambda group: group[
                        "score"
                    ],
                )

                alignments.append({
                    "suspicious_indices": (
                        suspicious_group[
                            "suspicious_indices"
                        ]
                    ),
                    "source_groups": [
                        best_source_group
                    ],
                    "score": (
                        best_source_group["score"]
                    ),
                })

            continue

        alignments.extend(best_paths)

    return alignments


def alignments_to_spans(
    alignments,
    suspicious_sentences,
    source_sentences,
):
    '''
    Convert alignment paths into character-level spans.

    Sentence indices are converted to character offsets
    using the sentence metadata.
    '''

    spans = []

    for alignment in alignments:

        suspicious_indices = alignment[
            "suspicious_indices"
        ]

        source_groups = alignment[
            "source_groups"
        ]

        if not suspicious_indices:
            continue

        if not source_groups:
            continue

        suspicious_start = (
            suspicious_sentences[
                suspicious_indices[0]
            ]["offset_start"]
        )

        suspicious_end = (
            suspicious_sentences[
                suspicious_indices[-1]
            ]["offset_end"]
        )


        source_start_idx = (
            source_groups[0]["source_start"]
        )

        source_end_idx = (
            source_groups[-1]["source_end"]
        )

        source_start = (
            source_sentences[
                source_start_idx
            ]["offset_start"]
        )

        source_end = (
            source_sentences[
                source_end_idx
            ]["offset_end"]
        )

        all_matches = []

        for source_group in source_groups:
            all_matches.extend(
                source_group["matches"]
            )

        spans.append({
            "suspicious_start": suspicious_start,
            "suspicious_length": (
                suspicious_end
                - suspicious_start
            ),

            "source_start": source_start,
            "source_length": (
                source_end
                - source_start
            ),

            "score": alignment["score"],

            "suspicious_indices": (
                suspicious_indices
            ),

            "source_groups": source_groups,

            "matches": all_matches,
        })

    return spans