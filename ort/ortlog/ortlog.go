package ortlog

import (
	"log"
	"os"
)

// Logger defines a structured logging interface (key-value style, compatible with zap/slog/logrus)
type Logger interface {
	Debugw(msg string, keysAndValues ...interface{})
	Infow(msg string, keysAndValues ...interface{})
	Warnw(msg string, keysAndValues ...interface{})
	Errorw(msg string, keysAndValues ...interface{})
}

var globalLogger Logger = newFallbackLogger()

// SetLogger replaces the global Logger (e.g. inject zap.SugaredLogger).
// Passing nil is ignored (keeps the current Logger).
func SetLogger(l Logger) {
	if l != nil {
		globalLogger = l
	}
}

// L returns the current global Logger.
func L() Logger {
	return globalLogger
}

// fallbackLogger is the built-in default implementation using the standard library log.
// Pre-creates 4 Logger instances to avoid heap allocation on each log call, reducing GC pressure.
type fallbackLogger struct {
	debugLog *log.Logger
	infoLog  *log.Logger
	warnLog  *log.Logger
	errorLog *log.Logger
}

func newFallbackLogger() *fallbackLogger {
	flags := log.LstdFlags
	return &fallbackLogger{
		debugLog: log.New(os.Stdout, "[DEBUG] [ort] ", flags),
		infoLog:  log.New(os.Stdout, "[INFO]  [ort] ", flags),
		warnLog:  log.New(os.Stdout, "[WARN]  [ort] ", flags),
		errorLog: log.New(os.Stderr, "[ERROR] [ort] ", flags),
	}
}

func (f *fallbackLogger) Debugw(msg string, keysAndValues ...interface{}) {
	f.debugLog.Println(append([]interface{}{msg}, keysAndValues...)...)
}

func (f *fallbackLogger) Infow(msg string, keysAndValues ...interface{}) {
	f.infoLog.Println(append([]interface{}{msg}, keysAndValues...)...)
}

func (f *fallbackLogger) Warnw(msg string, keysAndValues ...interface{}) {
	f.warnLog.Println(append([]interface{}{msg}, keysAndValues...)...)
}

func (f *fallbackLogger) Errorw(msg string, keysAndValues ...interface{}) {
	f.errorLog.Println(append([]interface{}{msg}, keysAndValues...)...)
}

func Debugw(msg string, keysAndValues ...interface{}) {
	globalLogger.Debugw(msg, keysAndValues...)
}

func Infow(msg string, keysAndValues ...interface{}) {
	globalLogger.Infow(msg, keysAndValues...)
}

func Warnw(msg string, keysAndValues ...interface{}) {
	globalLogger.Warnw(msg, keysAndValues...)
}

func Errorw(msg string, keysAndValues ...interface{}) {
	globalLogger.Errorw(msg, keysAndValues...)
}
