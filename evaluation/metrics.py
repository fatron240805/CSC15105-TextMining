import math

class Evaluation:
    '''
    Alignment evaluation for PAN given the following structure of ground truth and predictions:

    ground_truth = [
        {
            "this_offset": ...,
            "this_length": ...,
            "source_reference": ...,
            "source_offset": ...,
            "source_length": ...
        },
        ...
    ]

    predictions = [
        {
            "this_offset": ...,
            "this_length": ...,
            "source_reference": ...,
            "source_offset": ...,
            "source_length": ...
        },
        ...
    ]
    '''

    # Evaluation method given ground truth and predictions
    def evaluate(self, ground_truth, predictions):
        '''
        Input:
            ground_truth: List of dictionary ground truth plagiarism
            predictions: List of dictionary predicted plagiarism

        Output:
            Return dictionary of evaluation result 
        '''
        # True Positive, False Positive, False Negative 
        tp = self._true_positive(ground_truth, predictions)
        fp = self._false_positive(ground_truth, predictions, tp)
        fn = self._false_negative(ground_truth, predictions, tp)

        # Precision, Recall and F1 Score
        precision = self._precision(tp, fp)
        recall = self._recall(tp, fn)
        f1 = self._f1_score(precision, recall)

        # Granularity
        granularity = self._granularity(ground_truth, predictions)

        # Plagdet
        plagdet = self._plagdet(f1, granularity)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "granularity": granularity,
            "plagdet": plagdet
        }

    # Interval merging operation
    def _merge_intervals(self, intervals):
        '''
        Merge overlapping intervals
        For example:
            [(1, 20); (15, 35)] -> [(1, 35)]

        Input:
            intervals : list[(start, end)]
        
        Output:
            Return merged intervals
        '''
        if not intervals:
            return []

        intervals = sorted(intervals, key=lambda x: x[0])
        merged = [intervals[0]]

        for start, end in intervals[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))

        return merged

    # Union operation
    def _union_length(self, spans):
        '''
        Total covered characters
        For example:
            [
            {"this_offset": 5, "this_length": 20},
            {"this_offset": 15, "this_length": 30},
            {"this_offset": 50, "this_length": 10}
            ]
        Every single span turn into interval:
            [(5, 25), (15, 45), (50, 60)]
        -> Merge into [(5, 45), (50, 60)]
        -> Union length = (45 - 5) + (60 - 50) = 50
        '''
        intervals = []

        for span in spans:
            start = span["this_offset"]
            end = start + span["this_length"]
            intervals.append((start, end))

        merged = self._merge_intervals(intervals)

        return sum(
            end - start
            for start, end in merged
        )

    # Intersection operation
    def _intersection_length(self, gt_spans, pred_spans):
        '''
        Character-level overlap between ground truth and prediction
        '''
        gt_intervals = [
            (
                span["this_offset"],
                span["this_offset"] + span["this_length"]
            )
            for span in gt_spans
        ]

        pred_intervals = [
            (
                span["this_offset"],
                span["this_offset"] + span["this_length"]
            )
            for span in pred_spans
        ]

        gt_intervals = self._merge_intervals(gt_intervals)
        pred_intervals = self._merge_intervals(pred_intervals)

        i = 0
        j = 0
        overlap = 0

        while i < len(gt_intervals) and j < len(pred_intervals):

            gt_start, gt_end = gt_intervals[i]
            pred_start, pred_end = pred_intervals[j]

            start = max(gt_start, pred_start)
            end = min(gt_end, pred_end)

            if start < end:
                overlap += end - start

            if gt_end <= pred_end:
                i += 1
            else:
                j += 1

        return overlap

    def _span_overlap(self, span1, span2):
        '''
        Check whether two spans overlap.
        '''
        if span1["source_reference"] != span2["source_reference"]:
            return False

        start1 = span1["this_offset"]
        end1 = start1 + span1["this_length"]

        start2 = span2["this_offset"]
        end2 = start2 + span2["this_length"]

        return max(start1, start2) < min(end1, end2)

    
    # True Positve 
    def _true_positive(self, ground_truth, predictions):
        '''
        True Positive is the total length of character that detected to be plagiarism
        '''
        tp = 0
        sources = set()

        for span in ground_truth:
            sources.add(span["source_reference"])

        for span in predictions:
            sources.add(span["source_reference"])

        for source in sources:

            gt_source = [
                span
                for span in ground_truth
                if span["source_reference"] == source
            ]

            pred_source = [
                span
                for span in predictions
                if span["source_reference"] == source
            ]

            tp += self._intersection_length(
                gt_source,
                pred_source,
            )

        return tp

    # False Positive 
    def _false_positive(self, ground_truth, predictions, tp):
        '''
        False Positive is the total length of character that misdetected to be plagiarism
        '''
        predicted_character = self._union_length(predictions)
        return predicted_character - tp

    # False Negative 
    def _false_negative(self, ground_truth, predictions, tp):
        '''
        False Negative is the total length of characters that is not detected to be plagiarism but actually they are
        '''
        truth = self._union_length(ground_truth)
        return truth - tp

    # Precision metric
    def _precision(self, tp, fp):
        if tp + fp == 0:
            return 0.0
        return tp / (tp + fp)

    # Recall metric
    def _recall(self, tp, fn):
        if tp + fn == 0:
            return 0.0
        return tp / (tp + fn)

    # F1 Score metric
    def _f1_score(self, precision, recall):
        if precision + recall == 0:
            return 0.0
        return ((2 * precision * recall) / (precision + recall))

    # Granularity metric
    def _granularity(self, ground_truth, predictions):
        detected = []

        for gt_span in ground_truth:
            count = 0

            for pred_span in predictions:
                if self._span_overlap(gt_span, pred_span):
                    count += 1
            if count > 0:
                detected.append(count)

        if len(detected) == 0:
            return 1.0

        return sum(detected) / len(detected)

    # Plagdet metric
    def _plagdet(self, f1, granularity):
        return ((f1) / math.log2(1 + granularity))