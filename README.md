This is a benchmark developed by Tikeape, a member of TeichAI, designed to help identify how effectively a distilled model managed to mimic its style.

## Guide: Read Before Using!

When using the benchmark to compare a distilled model to a teacher model, I first suggest this to ensure you have a baseline.

At a temperature of 1.0 or above, have the teacher model compare a few times to itself, and take the average as the "Baseline accuracy" for that model. Then I suggest you run the benchmark at **MINIMUM 10 times** with the teacher and base model, and take the average. I would vary the topic and prompt.

Additionally, I suggest you compare the base model to a model of a completely different family to get a baseline of a "low" score, then use this to determine how good the score of the distilled model is. I'm sure this is some math calculations or system that can be used to dynamically calculate a similarity "elo" for all the models, to see which distill is closest to the base model, and I may attempt to implement that later.

An important note is that you can't really compare short things like greetings and 1–2 sentence chats with this, as those, even with the same model, are very different or practically exactly the same.

Another important note is that, even if you only compare proper essays from the models, if they are quite similar it most likely means the distill was successful overall.
