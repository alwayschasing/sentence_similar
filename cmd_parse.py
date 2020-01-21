import tensorflow as tf

flags = tf.flags


def cmd_parse():
    flags.DEFINE_string(
        "config_file",None,
        "configure file path"
    )
    flags.DEFINE_string(
        "input_files", None,
        "input_files path"
    )
    flags.DEFINE_string(
        "output_dir",None,
        "output dir"
    )
    flags.DEFINE_string(
        "init_checkpoint",None,
        "init checkpoint dir"
    )
    flags.DEFINE_integer(
        "batch_size",32,
        "batch size"
    )
    flags.DEFINE_integer(
        "save_summary_steps",10,
        "save_summary_steps"
    )
    flags.DEFINE_integer(
        "save_checkpoints_steps",3,
        "save_checkpoints_steps"
    )
    flags.DEFINE_integer(
        "keep_checkpoint_max",5,
        "keep_checkpoint_max"
    )
    flags.DEFINE_integer(
        "log_step_count_steps",10,
        "log_step_count_steps"
    )

    return flags.FLAGS