# author: ddukic

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb

api = wandb.Api()


def extract_target_performance(results, setting, metric="f1"):
    try:
        if setting.startswith("0_shot"):
            return results["test_all_metrics_zero_shot"]["identification"][
                "overall_" + metric
            ]
        else:
            if "test_all_metrics_few_shot_trigger_averaged" in results.keys():
                return results["test_all_metrics_few_shot_trigger_averaged"][
                    "overall_" + metric + "_averaged"
                ]
            elif "test_all_metrics_few_shot_averaged" in results.keys():
                return results["test_all_metrics_few_shot_averaged"][
                    "overall_" + metric + "_averaged"
                ]
            else:
                print("Probably a mistake for setting:", setting)
                return 0.0
    except:
        print("Probably a mistake for setting:", setting)
        return 0.0


def extract_name(run_url):
    try:
        run_key = run_url.split("/")[-1]
        return api.run("ddukic/oee-paper/" + run_key).name
    except:
        return ""


def transform_df(df):
    df.index.rename("setup", inplace=True)
    df = df.reset_index("setup")
    df["shots"] = df["setup"].apply(lambda x: int(re.findall(r"\d+", x)[0]))
    df["setup"] = (
        df["setup"]
        .apply(
            lambda x: x
            if x.startswith("0_shot")
            else x.replace("_" + re.findall(r"\d+", x)[0] + "_shot", "")
        )
        .apply(
            lambda x: x.replace(
                "_shot"
                if re.findall(r"\d+", x) == []
                else re.findall(r"\d+", x)[0] + "_shot",
                "only_target",
            )
        )
    )
    for setup in df.setup.unique():
        if "joint_pretrained" in setup:
            row_to_insert = df[df.setup == "only_target_pretrained_on_source"][
                ["vanilla", "implicit_multitask", "two_head_multitask", "shots"]
            ].to_dict("records")[0]
            row_to_insert["setup"] = setup
            df = pd.concat([df, pd.DataFrame([row_to_insert])], ignore_index=True)
        elif "seq_pretrained" in setup:
            row_to_insert = df[df.setup == "only_target_pretrained_on_source"][
                ["vanilla", "implicit_multitask", "two_head_multitask", "shots"]
            ].to_dict("records")[0]
            row_to_insert["setup"] = setup
            df = pd.concat([df, pd.DataFrame([row_to_insert])], ignore_index=True)
        elif setup == "joint_from_roberta":
            row_to_insert = df[df.setup == "only_target_from_roberta"][
                ["vanilla", "implicit_multitask", "two_head_multitask", "shots"]
            ].to_dict("records")[0]
            row_to_insert["setup"] = setup
            df = pd.concat([df, pd.DataFrame([row_to_insert])], ignore_index=True)

    df = df.drop(df[df.setup == "only_target_pretrained_on_source"].index)
    df = pd.melt(
        df,
        id_vars=["setup", "shots"],
        value_vars=["vanilla", "implicit_multitask", "two_head_multitask"],
        var_name="model",
        value_name="F1",
    )
    return df


def plot_graph(df):
    plt.rcParams["figure.dpi"] = 300
    sns.set_style("darkgrid")
    plt.yticks(np.arange(0, max(df["F1"]), 0.05))
    plt.xticks([0, 5, 10, 50, 100, 250, 500], rotation=45)
    for tick, size in zip(plt.xticks()[-1], [2, 3, 4, 5, 6, 7, 8]):
        tick.set_fontsize(size)
    sns.lineplot(
        df,
        x="shots",
        y="F1",
        hue="setup",
        style="model",
        alpha=0.6,
        palette=sns.color_palette("colorblind"),
    )
    plt.show()
