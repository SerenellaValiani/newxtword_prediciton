�'$	'94�c$�?3\o8b�?�2�gg?!�s`9B� @	!       "\
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsS"�^F�?�v��N#�?AZ��ڊ��?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails��/EH�?c	kc�w?A�ۡa1��?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails�o�DIH�?��e1���?A	�<��tz?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails��H�}}?�?�,u?A�j��P�`?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails	���W�?)�A&9�?A?rk�m��?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails�8F�G��?�w��?A�5�U�ũ?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCp\�M�?Eկt><�?A�0���?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails�@-s?�m�2;?A)<hv�[q?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails�2�gg?ٓ��<c?A��D-ͭ@?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails	�_ ���?�K�b�{?Aٯ;�y�?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails
5��,�?r���	�?Af���i�?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails�<dʇ��?�9:Z��?A�`��5�?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsƇ�˶��?���SV��?A�c�C��?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsbL�{)<�?��Gp#e�?AKZ����?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails����?��?V�1�Ұ?A���3���?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails��w��D�?Ҋo(|��?A�����a?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails�s`9B� @��!r�z�?AdϞ��d�?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails�h�wa�?1е/�w?A�\��ky�?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails��an�?yܝ�ۦ?A:"ߥ�%�?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails�G��|�?5�𤅫?A����?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails��E_A��?;�p�GR�?A[�����?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails>�*�?���@�m�?A�MG 7�g?*	'1�o@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate}���EC�?!�	CaתD@)Y"���?1���0��B@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat
�O��0�?!��kvx4>@)�~31]��?1��-l/�;@:Preprocessing2F
Iterator::Model��@�ȓ�?!C�N}6@)��N^�?1N���)0@:Preprocessing2U
Iterator::Model::ParallelMapV2z�]�zk�?!֏5���@)z�]�zk�?1֏5���@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��D�Ɵ�?!�{Z��`S@)G�@�]��?1m��k@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice"�^F�܂?!	7ӯ@)"�^F�܂?1	7ӯ@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorr��Q��z?!�d�QH�@)r��Q��z?1�d�QH�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapq:�V�?!�/����E@)��2�68q?1��d�	�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 18.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
$	�O�cM��?b�[�f�?�m�2;?!�9:Z��?	!       "	!       *	!       2$	V^i��_�?�}���?��D-ͭ@?!dϞ��d�?:	!       B	!       J	!       R	!       Z	!       JCPU_ONLYb Y      Y@q:����?"�
both�Your program is POTENTIALLY input-bound because 18.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: B 