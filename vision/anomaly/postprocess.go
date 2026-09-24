package anomaly

import (
	"fmt"
	"image"
	"math"
	"sort"

	ort "github.com/DavidSche/raven-onnxruntime/ort"
)

type regionComponent struct {
	box        image.Rectangle
	peakScore  float32
	meanScore  float32
	area       int
	sampleBase int
	sampleLen  int
	order      int
}

// postprocess converts one batch element from the strict two-output ONNX
// contract into AnomalyResult. Region coordinates are calculated on the raw
// ONNX map and then mapped back to the original ROI image.
func postprocess(outputValues map[string]*ort.Value, cfg Config, original preprocessResult) (*AnomalyResult, error) {
	if len(outputValues) != cfg.OutputContract.OutputCount {
		return nil, fmt.Errorf("%w: got %d outputs, want exactly %d", ErrInvalidContract, len(outputValues), cfg.OutputContract.OutputCount)
	}
	var scoreOutput, mapOutput *ort.Value
	for name, value := range outputValues {
		switch name {
		case "pred_score":
			scoreOutput = value
		case "anomaly_map":
			mapOutput = value
		default:
			return nil, fmt.Errorf("%w: unexpected output %q", ErrInvalidContract, name)
		}
	}
	if scoreOutput == nil || mapOutput == nil {
		return nil, fmt.Errorf("%w: pred_score and anomaly_map are both required", ErrInvalidContract)
	}

	scoreShape, err := scoreOutput.GetShape()
	if err != nil {
		return nil, fmt.Errorf("failed to get pred_score shape: %w", err)
	}
	if len(scoreShape) != 1 || scoreShape[0] != 1 {
		return nil, fmt.Errorf("%w: pred_score shape %v, want [1]", ErrInvalidContract, scoreShape)
	}
	scoreData, err := ort.GetTensorData[float32](scoreOutput)
	if err != nil {
		return nil, fmt.Errorf("failed to read pred_score: %w", err)
	}
	if len(scoreData) != 1 || !isFiniteFloat32(scoreData[0]) {
		return nil, fmt.Errorf("%w: pred_score must contain one finite value", ErrInvalidContract)
	}

	mapShape, err := mapOutput.GetShape()
	if err != nil {
		return nil, fmt.Errorf("failed to get anomaly_map shape: %w", err)
	}
	if len(mapShape) != 4 || mapShape[0] != 1 || mapShape[1] != 1 {
		return nil, fmt.Errorf("%w: anomaly_map shape %v, want [1,1,H,W]", ErrInvalidContract, mapShape)
	}
	mapHeight, mapWidth := int(mapShape[2]), int(mapShape[3])
	if mapHeight < 1 || mapWidth < 1 || mapHeight > math.MaxInt/mapWidth {
		return nil, fmt.Errorf("%w: anomaly_map dimensions are invalid", ErrInvalidContract)
	}
	mapData, err := ort.GetTensorData[float32](mapOutput)
	if err != nil {
		return nil, fmt.Errorf("failed to read anomaly_map: %w", err)
	}
	if int64(mapHeight)*int64(mapWidth) > int64(math.MaxInt) {
		return nil, fmt.Errorf("%w: anomaly_map dimensions overflow addressable pixels", ErrInvalidContract)
	}
	expectedPixels := mapHeight * mapWidth
	if len(mapData) != expectedPixels {
		return nil, fmt.Errorf("%w: anomaly_map has %d values, want %d", ErrInvalidContract, len(mapData), expectedPixels)
	}
	for _, value := range mapData {
		if !isFiniteFloat32(value) {
			return nil, fmt.Errorf("%w: anomaly_map contains a non-finite value", ErrInvalidContract)
		}
	}

	sourceMap := make([]float32, expectedPixels)
	copy(sourceMap, mapData)
	return postprocessData(scoreData[0], sourceMap, mapWidth, mapHeight, cfg, original)
}

func postprocessData(score float32, sourceMap []float32, mapWidth, mapHeight int, cfg Config, original preprocessResult) (*AnomalyResult, error) {
	mapDownsample := cfg.MapDownsample
	if mapDownsample <= 0 {
		mapDownsample = 1
	}
	if int64(mapHeight)*int64(mapWidth) > int64(math.MaxInt) {
		return nil, fmt.Errorf("%w: anomaly_map dimensions overflow addressable pixels", ErrInvalidContract)
	}
	expectedPixels := mapHeight * mapWidth
	if mapWidth < 1 || mapHeight < 1 || len(sourceMap) != expectedPixels {
		return nil, fmt.Errorf("%w: anomaly_map dimensions are invalid", ErrInvalidContract)
	}
	for _, value := range sourceMap {
		if !isFiniteFloat32(value) {
			return nil, fmt.Errorf("%w: anomaly_map contains a non-finite value", ErrInvalidContract)
		}
	}
	if !isFiniteFloat32(score) {
		return nil, fmt.Errorf("%w: pred_score must be finite", ErrInvalidContract)
	}
	var maxMap float32
	if expectedPixels > 0 {
		maxMap = sourceMap[0]
		for _, value := range sourceMap[1:] {
			if value > maxMap {
				maxMap = value
			}
		}
	}
	if cfg.ScoreContract.Semantics == ScoreSemanticsMapMax && cfg.ScoreContract.Assertion.Enabled {
		left := float64(score)
		right := float64(maxMap)
		scale := math.Max(math.Abs(left), 1e-12)
		if math.Abs(left-right) > cfg.ScoreContract.Assertion.Atol+cfg.ScoreContract.Assertion.Rtol*scale {
			return nil, fmt.Errorf("%w: pred_score %g differs from map max %g beyond the score contract tolerance", ErrInvalidContract, score, maxMap)
		}
	}

	components, thresholded, filteredByArea, rawComponentCount, samples, err := findRegionComponents(sourceMap, mapWidth, mapHeight, cfg.MapThreshold, cfg.MinRegionArea)
	if err != nil {
		return nil, err
	}
	type rankedRegion struct {
		region        Region
		qualification RegionQualificationStats
		order         int
	}
	ranked := make([]rankedRegion, 0, len(components))
	for _, component := range components {
		region := Region{
			PeakScore: component.peakScore,
			MeanScore: component.meanScore,
			Area:      component.area,
			Coverage:  float32(component.area) / float32(expectedPixels),
		}
		region.Box = mapModelBox(component.box, mapWidth, mapHeight, original.originalWidth, original.originalHeight)
		sampled := samples[component.sampleBase : component.sampleBase+component.sampleLen]
		sorted := make([]float32, len(sampled))
		copy(sorted, sampled)
		sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
		ranked = append(ranked, rankedRegion{
			region: region,
			qualification: RegionQualificationStats{
				RegionIndex:   0,
				ScoreP90:      percentile(sorted, 0.90),
				ScoreP95:      percentile(sorted, 0.95),
				SampledScores: append([]float32(nil), sampled...),
			},
			order: component.order,
		})
	}

	debugStats := AnomalyDebugStats{
		RawComponentCount:     rawComponentCount,
		ThresholdedPixelCount: thresholded,
		FilteredByArea:        filteredByArea,
	}
	if len(ranked) > cfg.MaxRegions {
		debugStats.FilteredByMaxRegions = len(ranked) - cfg.MaxRegions
		ranked = ranked[:cfg.MaxRegions]
	}

	sort.SliceStable(ranked, func(i, j int) bool {
		if ranked[i].region.PeakScore != ranked[j].region.PeakScore {
			return ranked[i].region.PeakScore > ranked[j].region.PeakScore
		}
		if ranked[i].region.Area != ranked[j].region.Area {
			return ranked[i].region.Area > ranked[j].region.Area
		}
		return ranked[i].region.Coverage > ranked[j].region.Coverage
	})
	regions := make([]Region, len(ranked))
	qualification := make([]RegionQualificationStats, len(ranked))
	for index, item := range ranked {
		regions[index] = item.region
		item.qualification.RegionIndex = index
		qualification[index] = item.qualification
	}
	debugStats.Regions = qualification

	result := &AnomalyResult{
		Score:                 score,
		ScoreSemantics:        cfg.ScoreSemantics,
		IsAnomalous:           score >= cfg.ScoreThreshold,
		RegionDecisionMode:    cfg.RegionDecisionMode,
		MapSourceWidth:        mapWidth,
		MapSourceHeight:       mapHeight,
		MapDownsample:         mapDownsample,
		Regions:               regions,
		DebugStats:            debugStats,
		ThresholdVersion:      cfg.ThresholdVersion,
		RegionPolicyVersion:   cfg.RegionPolicyVersion,
		DecisionPolicyVersion: cfg.DecisionPolicyVersion,
	}

	if cfg.OutputAnomalyMap {
		outputMap := downsampleMap(sourceMap, mapWidth, mapHeight, mapDownsample)
		result.AnomalyMap = outputMap
		result.MapHeight = (mapHeight + mapDownsample - 1) / mapDownsample
		result.MapWidth = (mapWidth + mapDownsample - 1) / mapDownsample
	}
	return result, nil
}

func findRegionComponents(mapData []float32, width, height int, threshold float32, minRegionArea int) ([]regionComponent, int, int, int, []float32, error) {
	pixels := width * height
	labels := make([]int32, pixels)
	stack := make([]int, 0, min(1024, pixels))
	components := make([]regionComponent, 0, 16)
	samples := make([]float32, 0, pixels)
	thresholded := 0
	componentID := int32(0)

	for start := 0; start < pixels; start++ {
		if labels[start] != 0 || mapData[start] < threshold {
			continue
		}
		componentID++
		if componentID == math.MaxInt32 {
			return nil, 0, 0, 0, nil, fmt.Errorf("%w: too many anomaly components", ErrInvalidContract)
		}
		stack = append(stack[:0], start)
		labels[start] = componentID
		minX, minY, maxX, maxY := width, height, -1, -1
		area := 0
		peak := float32(math.Inf(-1))
		sum := float64(0)
		base := len(samples)

		for len(stack) > 0 {
			index := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			x, y := index%width, index/width
			if x < minX {
				minX = x
			}
			if x > maxX {
				maxX = x
			}
			if y < minY {
				minY = y
			}
			if y > maxY {
				maxY = y
			}
			value := mapData[index]
			samples = append(samples, value)
			area++
			thresholded++
			sum += float64(value)
			if value > peak {
				peak = value
			}

			for dy := -1; dy <= 1; dy++ {
				neighborY := y + dy
				if neighborY < 0 || neighborY >= height {
					continue
				}
				for dx := -1; dx <= 1; dx++ {
					neighborX := x + dx
					if neighborX < 0 || neighborX >= width {
						continue
					}
					neighbor := neighborY*width + neighborX
					if labels[neighbor] != 0 || mapData[neighbor] < threshold {
						continue
					}
					labels[neighbor] = componentID
					stack = append(stack, neighbor)
				}
			}
		}

		components = append(components, regionComponent{
			box:        image.Rect(minX, minY, maxX+1, maxY+1),
			peakScore:  peak,
			meanScore:  float32(sum / float64(area)),
			area:       area,
			sampleBase: base,
			sampleLen:  area,
			order:      int(componentID),
		})
	}

	kept := components[:0]
	filtered := 0
	for _, component := range components {
		if component.area >= minRegionArea {
			kept = append(kept, component)
		} else {
			filtered++
		}
	}
	return kept, thresholded, filtered, int(componentID), samples, nil
}

func mapModelBox(box image.Rectangle, modelWidth, modelHeight, originalWidth, originalHeight int) image.Rectangle {
	if originalWidth <= 0 || originalHeight <= 0 {
		return image.Rectangle{}
	}
	scaleX := float64(originalWidth) / float64(modelWidth)
	scaleY := float64(originalHeight) / float64(modelHeight)
	minX := int(math.Floor(float64(box.Min.X) * scaleX))
	minY := int(math.Floor(float64(box.Min.Y) * scaleY))
	maxX := int(math.Ceil(float64(box.Max.X) * scaleX))
	maxY := int(math.Ceil(float64(box.Max.Y) * scaleY))
	minX = clampInt(minX, 0, originalWidth)
	minY = clampInt(minY, 0, originalHeight)
	maxX = clampInt(maxX, 0, originalWidth)
	maxY = clampInt(maxY, 0, originalHeight)
	if minX >= maxX {
		maxX = minX + 1
	}
	if minY >= maxY {
		maxY = minY + 1
	}
	if maxX > originalWidth {
		maxX = originalWidth
	}
	if maxY > originalHeight {
		maxY = originalHeight
	}
	if minX >= maxX || minY >= maxY {
		return image.Rectangle{}
	}
	return image.Rect(minX, minY, maxX, maxY)
}

func downsampleMap(source []float32, width, height, factor int) []float32 {
	if factor <= 1 {
		output := make([]float32, len(source))
		copy(output, source)
		return output
	}
	outputWidth := (width + factor - 1) / factor
	outputHeight := (height + factor - 1) / factor
	output := make([]float32, outputWidth*outputHeight)
	for y := 0; y < outputHeight; y++ {
		sourceY0, sourceY1 := y*factor, min((y+1)*factor, height)
		for x := 0; x < outputWidth; x++ {
			sourceX0, sourceX1 := x*factor, min((x+1)*factor, width)
			sum := float64(0)
			count := 0
			for sourceY := sourceY0; sourceY < sourceY1; sourceY++ {
				for sourceX := sourceX0; sourceX < sourceX1; sourceX++ {
					sum += float64(source[sourceY*width+sourceX])
					count++
				}
			}
			output[y*outputWidth+x] = float32(sum / float64(count))
		}
	}
	return output
}

func percentile(sorted []float32, probability float64) float32 {
	if len(sorted) == 0 {
		return 0
	}
	index := int(math.Ceil(probability*float64(len(sorted)))) - 1
	index = clampInt(index, 0, len(sorted)-1)
	return sorted[index]
}

func isFiniteFloat32(value float32) bool {
	return !math.IsNaN(float64(value)) && !math.IsInf(float64(value), 0)
}

func clampInt(value, minimum, maximum int) int {
	if value < minimum {
		return minimum
	}
	if value > maximum {
		return maximum
	}
	return value
}
