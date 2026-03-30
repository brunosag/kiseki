<script lang="ts">
	import { getCheckpoints } from './data.remote';
	import LoaderCircle from '@lucide/svelte/icons/loader-circle';
</script>

<main class="mx-auto flex h-full max-w-160 flex-col gap-16 pt-24 font-mono">
	{#await getCheckpoints()}
		<div class="flex grow items-center justify-center">
			<LoaderCircle class="animate-spin text-zinc-500" />
		</div>
	{:then checkpoints}
		<div>
			<h2 class="mb-0.5 text-center text-2xl text-zinc-600 uppercase">Load checkpoint</h2>
			<!-- <p class="text-zinc-500">Please select a model checkpoint to analyse.</p> -->
		</div>
		<div class="flex flex-col gap-4">
			{#each checkpoints as { id, optimizer, best_accuracy, timestamp, iteration, batchsize, hyperparameters }}
				<a
					href="/{id}"
					target="_blank"
					class="w-full border-2 border-zinc-800 px-8 py-6 hover:bg-zinc-800/40 active:bg-zinc-800"
				>
					<div class="mb-2 flex items-center justify-between">
						<span class="text-lg">{optimizer}</span>
						<div class="flex items-baseline gap-1">
							<span class="text-sm font-bold text-zinc-600">i</span>
							<span class="font-bold text-zinc-400">{iteration}</span>
						</div>
						<div class="flex items-baseline gap-1">
							<span class="text-sm font-bold text-zinc-600">acc</span>
							<span class="font-bold text-zinc-400">
								{best_accuracy.toFixed(2)}%
							</span>
						</div>
						<span class="text-sm text-zinc-700">{timestamp}</span>
					</div>
					<div class="flex flex-wrap gap-x-8 gap-y-0.5 text-sm">
						<div class="flex gap-1">
							<span class="text-zinc-700">batchsize</span>
							<span class="text-zinc-600">{batchsize}</span>
						</div>
						{#each Object.entries(hyperparameters) as [key, value]}
							<div class="flex gap-1">
								<span class="text-zinc-700">{key}</span>
								<span class="text-zinc-600">{value}</span>
							</div>
						{/each}
					</div>
				</a>
			{/each}
		</div>
	{/await}
</main>
