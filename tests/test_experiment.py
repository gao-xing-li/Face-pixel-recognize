import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from face_pixel.data_io import read_json, write_csv, write_json, read_binary_mask, read_dataset_manifest
from face_pixel.dataset import binary_labels, prepare_dataset
from face_pixel.experiment import run_experiment
from face_pixel.validation import predict, settings


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rows = []
        for i, split in enumerate(('train', 'val', 'test', 'test')):
            image = np.full((7, 7, 3), [255, 0, 0], dtype=np.uint8)
            label = np.ones((7, 7), dtype=np.uint8)
            label[0, 0] = 3
            Image.fromarray(image).save(self.root / f'image{i}.png')
            Image.fromarray(label).save(self.root / f'label{i}.png')
            rows.append({'sample_id': f'sample{i}', 'split': split, 'image_path': f'image{i}.png', 'label_path': f'label{i}.png'})
        write_csv(self.root / 'raw.csv', rows)
        prepare_dataset(self.root / 'raw.csv', self.root / 'prepared', self.root)
        self.manifest = self.root / 'prepared/manifest.csv'
        self.params = self.root / 'params.json'
        write_json(self.params, {'mean_cr': 255, 'std_cr': 1, 'mean_cb': 85, 'std_cb': 1})
        self.configs = []
        for name, method in (('original', 'none'), ('final', 'largest_fill')):
            path = self.root / f'{name}.json'
            write_json(path, {'config_id': name, **settings(1.25, method)})
            self.configs.append(path)

    def test_label_mapping_and_original_split(self):
        face, valid = binary_labels(np.arange(19, dtype=np.uint8).reshape(1, 19))
        self.assertEqual(np.flatnonzero(face).tolist(), [1, 2, 4, 5, 6, 7, 10, 11, 12])
        self.assertEqual(np.flatnonzero(valid == 0).tolist(), [3])
        rows = read_dataset_manifest(self.manifest)
        self.assertEqual([r['split'] for r in rows], ['train', 'validation', 'test', 'test'])
        self.assertEqual(read_binary_mask(rows[0]['valid_mask_path'])[0, 0], 0)

    def test_combined_detection_evaluation_and_visualization(self):
        output = self.root / 'experiment'
        result = run_experiment(self.manifest, output, configs=self.configs, params=self.params)
        for score in result['results'].values():
            self.assertEqual(score['global']['f1'], 1)
        self.assertEqual(result['comparison']['test']['f1'], 0)
        self.assertEqual(len(list((output / 'predictions/test-final/predictions').glob('*.png'))), 2)
        self.assertEqual(read_json(output / 'experiment.json'), result)
        from face_pixel.visualization import _collect
        entries, summary = _collect(self.manifest, output, 'original', 'final')
        self.assertEqual(len(entries), 2)
        self.assertEqual(summary['results']['test-final']['global']['f1'], 1)

    def test_old_version_metadata_does_not_block_prediction(self):
        config = read_json(self.configs[0])
        config.update(runtime={'python': '0.0', 'numpy': 'different'}, source_sha256={'detection.py': 'old'})
        write_json(self.configs[0], config)
        result = predict(self.manifest, self.configs[0], 'test', self.root / 'prediction', self.params)
        self.assertEqual(result['n_images'], 2)

    def test_prediction_does_not_read_truth(self):
        import face_pixel.validation as validation
        original_reader = validation.read_bgr
        with patch('face_pixel.validation.read_binary_mask', side_effect=AssertionError('Truth read during prediction')), \
             patch('face_pixel.validation.read_bgr', wraps=original_reader):
            predict(self.manifest, self.configs[0], 'test', self.root / 'prediction', self.params)

    def test_wrong_size_or_unknown_labels_are_rejected(self):
        with self.assertRaises(ValueError):
            binary_labels(np.array([[19]], dtype=np.uint8))
        with self.assertRaises(ValueError):
            binary_labels(np.array([[1.5]]))
        Image.new('L', (2, 2), 1).save(self.root / 'label0.png')
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            prepare_dataset(self.root / 'raw.csv', self.root / 'bad', self.root)


if __name__ == '__main__':
    unittest.main()
