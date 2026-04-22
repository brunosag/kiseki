<script lang="ts">
	import Math from '$lib/components/Math.svelte';
	import { Button } from '$lib/components/ui/button';
	import * as Field from '$lib/components/ui/field';
	import { Input } from '$lib/components/ui/input';
	import * as InputGroup from '$lib/components/ui/input-group';
	import * as RadioGroup from '$lib/components/ui/radio-group';
	import { Spinner } from '$lib/components/ui/spinner';
	import {
		ExperimentSocket,
		type BaseExperimentConfig,
		type ExperimentConfig,
		type OptimizerConfig
	} from './ExperimentSocket.svelte';

	// these options will come from either the backend or an external file
	const params = {
		leea: [
			{
				key: 'N',
				label: String.raw`N`,
				value: 200,
				description: 'Population size',
				type: 'number'
			},
			{
				key: 'pₘ',
				label: String.raw`p_{\mathrm{m}}`,
				description: 'Mutation probability',
				value: 0.04,
				type: 'number',
				step: 0.01
			},
			{
				key: 'η₀',
				label: String.raw`\eta_0`,
				value: 0.03,
				description: 'Initial mutation step size',
				type: 'number',
				step: 0.01
			},
			{
				key: 'γ',
				label: String.raw`\gamma`,
				value: 0.99,
				description: 'Mutation decay factor',
				type: 'number',
				step: 0.01
			},
			{
				key: 'ρ',
				label: String.raw`\rho`,
				value: 0.4,
				description: 'Retention fraction',
				type: 'number',
				step: 0.01
			},
			{
				key: 'ρₓ',
				label: String.raw`\rho_{\mathrm{x}}`,
				value: 0.5,
				description: 'Crossover fraction',
				type: 'number',
				step: 0.01
			},
			{
				key: 'λ',
				label: String.raw`\lambda`,
				value: 0.2,
				description: 'Fitness decay coefficient',
				type: 'number',
				step: 0.01
			},
			{
				key: 'τ_pat',
				label: String.raw`\tau_{\mathrm{pat}}`,
				value: 25,
				description: 'Validation patience threshold',
				type: 'number'
			}
		],
		sgd: [{ key: 'η', label: String.raw`\eta`, value: 0.01, description: 'Learning rate' }]
	};

	const socket = new ExperimentSocket('ws://127.0.0.1:8080/experiment');

	let selectedOptimizer = $state<'leea' | 'sgd'>('leea');
	let optConfig = $state<OptimizerConfig>({
		leea: {
			type: 'leea',
			N: 200,
			pₘ: 0.04,
			'η₀': 0.03,
			γ: 0.99,
			ρ: 0.4,
			ρₓ: 0.5,
			λ: 0.2,
			τ_pat: 25
		},
		sgd: { type: 'sgd', η: 0.01 }
	});
	let baseConfig = $state<BaseExperimentConfig>({
		dataset: 'mnist',
		device: 'gpu',
		seed: 42,
		batchsize: 1000,
		max_i: 100000,
		target_acc: 100.0
	});

	let config = $derived<ExperimentConfig>({
		...baseConfig,
		opt: optConfig[selectedOptimizer]
	});

	function startExperiment() {
		socket.startExperiment(config);
	}
</script>

{#snippet radio(name: string, options: { label: string; value: string }[])}
	{#each options as { label, value }}
		{@const id = `${name}-${value}`}
		<Field.Field class="h-9 w-fit" orientation="horizontal">
			<RadioGroup.Item {value} {id} />
			<Field.Label for={id} class="grow-0! cursor-pointer font-normal">{label}</Field.Label>
		</Field.Field>
	{/each}
{/snippet}

{#if socket.isLoading}
	<div class="flex h-full items-center justify-center">
		<Spinner class="size-8" />
	</div>
{:else if socket.isTraining}
	<div class="font-mono">
		<div>i: {socket.i}</div>
		<div>best_acc: {socket.bestAcc}</div>
		<div>trace: {JSON.stringify(socket.history, null, 2)}</div>
	</div>
{:else}
	<form class="mx-auto max-w-xl" onsubmit={startExperiment}>
		<Field.Group>
			<Field.Set class="grid grid-cols-[max-content_auto] gap-x-12! gap-y-3!">
				<Field.Field class="contents">
					<Field.Label>Dataset</Field.Label>
					<RadioGroup.Root class="md:flex md:gap-12" bind:value={baseConfig.dataset}>
						{@render radio('dataset', [
							{ label: 'MNIST', value: 'mnist' },
							{ label: 'Fashion MNIST', value: 'fashion' },
							{ label: 'CIFAR-10', value: 'cifar10' }
						])}
					</RadioGroup.Root>
				</Field.Field>

				<Field.Field class="contents">
					<Field.Label>Device</Field.Label>
					<RadioGroup.Root class="md:flex md:gap-12" bind:value={baseConfig.device}>
						{@render radio('device', [
							{ label: 'GPU', value: 'gpu' },
							{ label: 'CPU', value: 'cpu' }
						])}
					</RadioGroup.Root>
				</Field.Field>

				<Field.Field class="contents">
					<Field.Label class="whitespace-nowrap" for="seed">Seed</Field.Label>
					<Input class="w-24!" id="seed" type="number" bind:value={baseConfig.seed} />
				</Field.Field>

				<Field.Field class="contents">
					<Field.Label class="whitespace-nowrap" for="batchsize">Batch size</Field.Label>
					<Input
						class="w-24!"
						id="batchsize"
						type="number"
						bind:value={baseConfig.batchsize}
					/>
				</Field.Field>

				<Field.Field class="contents">
					<Field.Label class="whitespace-nowrap" for="iterations">Iterations</Field.Label>
					<Input
						class="w-24!"
						id="iterations"
						type="number"
						bind:value={baseConfig.max_i}
					/>
				</Field.Field>

				<Field.Field class="contents">
					<Field.Label class="whitespace-nowrap" for="target-acc"
						>Target accuracy</Field.Label
					>
					<InputGroup.Root class="w-24!">
						<InputGroup.Input
							id="target-acc"
							type="number"
							bind:value={baseConfig.target_acc}
						/>
						<InputGroup.Addon align="inline-end">
							<InputGroup.Text>%</InputGroup.Text>
						</InputGroup.Addon>
					</InputGroup.Root>
				</Field.Field>
			</Field.Set>

			<Field.Separator />

			<Field.Field class="gap-2">
				<Field.Label class="w-fit!">Optimizer</Field.Label>
				<RadioGroup.Root class="md:flex md:gap-12" bind:value={selectedOptimizer}>
					{@render radio('optimizer', [
						{ label: 'LEEA', value: 'leea' },
						{ label: 'SGD', value: 'sgd' }
					])}
				</RadioGroup.Root>
			</Field.Field>

			<Field.Set class="gap-3">
				<Field.Group class="grid grid-cols-[max-content_auto_1fr] items-center gap-3">
					{#each params[selectedOptimizer] as { key, label, description, ...rest }}
						<Field.Field class="contents">
							<Field.Label for="param-{label}">
								<Math math={label} />
							</Field.Label>
							<Input
								{...rest}
								class="w-20!"
								id="param-{label}"
								bind:value={(optConfig as any)[selectedOptimizer][key]}
								required
							/>
							<Field.Description>{description}</Field.Description>
						</Field.Field>
					{/each}
				</Field.Group>
			</Field.Set>

			<Field.Field orientation="responsive">
				<Button type="submit">Start</Button>
			</Field.Field>
		</Field.Group>
	</form>
{/if}
